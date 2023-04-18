from dataclasses import asdict, dataclass
from itertools import count
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from jinja2 import Environment, FileSystemLoader, Template

from .backend import Backend, MinichainContext, PromptSnap, RunLog

Input = TypeVar("Input")
Output = TypeVar("Output")
FnOutput = TypeVar("FnOutput")


@dataclass
class History:
    expand: Callable[[List[Any]], Iterator[Any]]
    inputs: List[Any]


@dataclass
class Break:
    pass


@dataclass
class Chain:
    history: History
    name: str
    cache: Any = None

    def run_gen(self) -> Any:
        # Lazily instantiate all the inputs
        args = []
        for i, base_input in enumerate(self.history.inputs):
            function_input = base_input
            if isinstance(base_input, Chain):
                if base_input.cache is not None:
                    function_input = base_input.cache
                    if isinstance(function_input, Break):
                        yield Break()
                        return
                else:
                    for function_input in base_input.run_gen():
                        if isinstance(function_input, Break):
                            base_input.cache = Break()
                            yield Break()
                            return
                        yield None

                    base_input.cache = function_input
            args.append(function_input)
        # Run the current prompt
        for out in self.history.expand(*args):
            if isinstance(out, Break):
                yield Break()
                return

            yield None
        yield out

    def run(self) -> Any:
        for x in self.run_gen():
            pass
        return x


class Prompt(Generic[Input, Output, FnOutput]):
    counter = count()

    def __init__(
        self,
        fn: Callable[[Callable[[Input], Output]], Iterable[FnOutput]],
        backend: Union[List[Backend], Backend],
        template_file: Optional[str],
        template: Optional[str],
        gradio_conf: Any = None,
    ):
        self.fn = fn
        if not isinstance(backend, List):
            self.backend = [backend]
        else:
            self.backend = backend

        self.template_file: Optional[str] = template_file
        self.template: Optional[str] = template
        self.gradio_conf = gradio_conf

        self._fn: str = fn.__name__
        self._id: int = Prompt.counter.__next__()

    def run(self, request: str, tool_num: int = 0) -> Iterator[RunLog]:
        if not hasattr(self.backend[tool_num], "run_stream"):
            yield RunLog(request, None)
            response: Union[str, Any] = self.backend[tool_num].run(request)
            yield RunLog(request, response)
        else:
            yield RunLog(request, None)
            for r in self.backend[tool_num].run_stream(request):
                yield RunLog(request, r)

    def template_fill(self, inp: Any) -> str:
        kwargs = inp
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = Template(self.template)

        return str(tmp.render(**kwargs))

    def __call__(self, *args: Any) -> Chain:
        return Chain(History(self.expand, list(args)), self.fn.__name__)

    class Model:
        def __init__(self, prompt: "Prompt[Input, Output, FnOutput]", data: Any):
            self.prompt = prompt
            self.data = data
            self.run_log = RunLog()

        def __call__(self, model_input: Any, tool_num: int = 0) -> Any:
            for r in self.stream(model_input, tool_num):
                yield r

            # print("hello tool")
            # for out in self.prompt.dynamic[tool_num].expand(*model_input):
            #     self.run_log = self.prompt.dynamic[tool_num].model.run_log
            #     self.run_log.dynamic = tool_num
            #     yield out

        def stream(
            self, model_input: Any, tool_num: int = 0
        ) -> Iterator[Optional[str]]:
            if (
                self.prompt.template is not None
                or self.prompt.template_file is not None
            ):
                if not isinstance(model_input, dict):
                    model_input = asdict(model_input)
                result = self.prompt.template_fill(model_input)
            else:
                result = model_input

            for run_log in self.prompt.run(result, tool_num):
                r = self.run_log.response
                if run_log.response is None:
                    out = r
                elif not r:
                    out = run_log.response
                else:
                    out = r + run_log.response
                self.run_log = RunLog(run_log.request, out, dynamic=tool_num)
                yield self.run_log.response

    def expand(
        self, *args: List[Any], data: Any = None
    ) -> Iterator[Optional[FnOutput]]:
        # Times prompt has been used.
        MinichainContext.prompt_count.setdefault(self._id, -1)
        MinichainContext.prompt_count[self._id] += 1
        count = MinichainContext.prompt_count[self._id]

        # Snap of the prompt
        MinichainContext.prompt_store.setdefault((self._id, count), [])
        MinichainContext.prompt_store[self._id, count].append(PromptSnap())

        # Model to be passed to function
        model = self.Model(self, data)
        for output in self.fn(model, *args):
            t = model.run_log
            assert model.run_log, str(model)
            snap = PromptSnap(args, t, output)
            count = MinichainContext.prompt_count[self._id]
            MinichainContext.prompt_store.setdefault((self._id, count), [])
            MinichainContext.prompt_store[self._id, count][-1] = snap
            yield None

        assert model.run_log, str(model)
        t = model.run_log
        snap = PromptSnap(args, t, output)
        MinichainContext.prompt_store[self._id, count][-1] = snap
        yield output


def prompt(
    backend: List[Backend] = [],
    template_file: Optional[str] = None,
    template: Optional[str] = None,
    gradio_conf: Optional[Any] = None,
) -> Callable[[Any], Prompt[Input, Output, FnOutput]]:
    return lambda fn: Prompt(fn, backend, template_file, template, gradio_conf)


def transform():  # type: ignore
    return lambda fn: lambda *args: Chain(
        History(lambda *x: iter((fn(*x),)), list(args)), fn.__name__
    )
