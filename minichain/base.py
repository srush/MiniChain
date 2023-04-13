import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from itertools import count
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from eliot import start_action
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)

from .backend import Backend, MinichainContext, Request

env = Environment(
    loader=PackageLoader("minichain"),
    autoescape=select_autoescape(),
    extensions=["jinja2_highlight.HighlightExtension"],
)


def _prompt(r: Union[str, Request]) -> Request:
    if isinstance(r, str):
        return Request(r)
    else:
        return r


Input = TypeVar("Input")
Output = TypeVar("Output")
FnOutput = TypeVar("FnOutput")


def enum(x: Type[Enum]) -> Dict[str, int]:
    d = {e.name: e.value for e in x}
    return d


def walk(x: Any) -> Any:
    if issubclass(x if get_origin(x) is None else get_origin(x), List):
        return {"_t_": "list", "t": walk(get_args(x)[0])}
    if issubclass(x, Enum):
        return enum(x)

    if is_dataclass(x):
        return {y.name: walk(y.type) for y in fields(x)}
    return x.__name__


def type_to_prompt(out: type) -> str:
    tmp = env.get_template("type_prompt.pmpt.tpl")
    d = walk(out)
    return tmp.render({"typ": d})


def simple(model, **kwargs):  # type: ignore
    return model(kwargs)


@dataclass
class History:
    prompt: "Prompt[Input, Output, FnOutput]"
    inputs: List[Any]


@dataclass
class Fail:
    arg_num: int
    data: Any

    def __str__(self):
        return "fail"


@dataclass
class Chain:
    # TODO: Add caching
    history: History

    def run_gen(self, trial: int = 0, data: Any = None) -> Any:
        args = []
        count = []
        for inp in self.history.inputs:
            inp2 = inp
            if isinstance(inp, Chain):
                for inp2 in inp.run_gen():
                    yield None
            args.append(inp2)
            count.append(0)
        for out in self.history.prompt.expand(args, trial, data):
            yield None
        while isinstance(out, Fail):
            count[out.arg_num] += 1
            inp = self.history.inputs[out.arg_num]
            assert isinstance(inp, Chain)
            for inp2 in inp.run_gen(trial=count[out.arg_num], data=out.data):
                yield None
            args[out.arg_num] = inp2
            for out in self.history.prompt.expand(args, trial, data):
                yield None
            yield None

        yield out

    def run(self) -> Any:
        for x in self.run_gen():
            if x is not None:
                return x


class Prompt(Generic[Input, Output, FnOutput]):
    counter = count()

    def __init__(
        self,
        backend: Backend,
        parser: Union[str, Callable[[str], Output]],
        template_file: Optional[str],
        template: Optional[str],
        stop_template: Optional[str],
        fn: Callable[[Callable[[Input], Output]], FnOutput] = simple,
        stream: bool = False,
        block_input=None,
        block_output=None,
    ):
        self.backend: Backend = backend
        self.stream = stream
        self.description: str = (
            backend.description if hasattr(backend, "description") else ""
        )
        self.parser: Union[str, Callable[[str], Output]] = parser
        self.template_file: Optional[str] = template_file
        self.template: Optional[str] = template
        self.stop_template: Optional[str] = stop_template
        self.block_input = block_input
        self.block_output = block_output
        self.fn = fn
        self._fn: str = fn.__name__
        self._id: int = Prompt.counter.__next__()

    def parse(self, response: str) -> Any:
        """
        Convert from the string response of the function
        to the output type.
        """
        if isinstance(self.parser, str):
            if self.parser == "str":
                return response
            elif self.parser == "json":
                return json.loads(response)
        else:
            return self.parser(response)

    def run_verbose(
        self, r: Union[str, Request], stream=False
    ) -> Tuple[Request, str, Output]:
        with start_action(action_type=str(self.fn)):
            request = _prompt(r)
            with start_action(action_type="Prompted", prompt=request.prompt):
                response: Union[str, Any] = self.backend.run(
                    request
                    if hasattr(self.backend, "needs_request")
                    else request.prompt
                )
            with start_action(action_type="Response", result=response):
                output = self.parse(response)
        if not isinstance(response, str):
            response = "(data)"
        return (request, response, output)

    def run_stream(
        self, r: Union[str, Request], stream=False
    ) -> Tuple[Request, str, Output]:
        request = _prompt(r)
        for r in self.backend.run_stream(
            request if hasattr(self.backend, "needs_request") else request.prompt
        ):
            yield (request, r, r)

    def template_fill(self, inp: Any) -> Request:
        kwargs = inp
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = Template(self.template)

        x = tmp.render(**kwargs)

        if self.stop_template:
            stop = [Template(self.stop_template).render(**kwargs)]
        else:
            stop = None
        return Request(x, stop)

    def __call__(self, *args: Any) -> FnOutput:
        return Chain(History(self, args))

    class Model:
        def __init__(self, prompt, trial, data):
            self.prompt = prompt
            self.trial = trial
            self.data = data
            self.run_log = None

        def fail(self, argnum: int, data: Any = None) -> Fail:
            return Fail(argnum - 1, data)

        def __call__(self, input_):
            for r in self.stream(input_):
                pass
            return r

        def stream(self, input_):
            assert self.run_log is None, "Only call `model` once per function"
            if (
                self.prompt.template is not None
                or self.prompt.template_file is not None
            ):
                if not isinstance(input_, dict):
                    input_ = asdict(input_)
                input_ = dict(**input_)
                input_["_trial"] = self.trial
                input_["_fail_data"] = self.data

                result = self.prompt.template_fill(input_)
            else:
                result = input_
            if self.prompt.stream and hasattr(self.prompt.backend, "run_stream"):
                self.run_log = ("", "", "")
                for a, b, c in self.prompt.run_stream(result):
                    self.run_log = (a, self.run_log[1] + b, self.run_log[2] + c)
                    yield c
            else:
                self.run_log = (Request(input_), "", "")
                yield None
                self.run_log = self.prompt.run_verbose(result)
                yield self.run_log[-1]

    def expand(self, args, trial=0, data=None):
        MinichainContext.prompt_count.setdefault(self._id, -1)
        if trial == 0:
            MinichainContext.prompt_count[self._id] += 1
        count = MinichainContext.prompt_count[self._id]
        MinichainContext.prompt_store.setdefault((self._id, count), [])
        MinichainContext.prompt_store[self._id, count].append(("", Request(""), "", ""))

        model = self.Model(self, trial, data)

        if self.stream:
            for output in self.fn(model, *args):

                t = model.run_log
                assert model.run_log, str(model)
                run_log = (args, t[0], t[1], output)
                count = MinichainContext.prompt_count[self._id]
                MinichainContext.prompt_store.setdefault((self._id, count), [])
                MinichainContext.prompt_store[self._id, count][-1] = run_log
                yield None
        else:
            output = self.fn(model, *args)

        if not isinstance(output, Fail):
            assert model.run_log, str(model)
            t = model.run_log
            run_log = (args, t[0], t[1], output)
            MinichainContext.prompt_store[self._id, count][-1] = run_log

        yield output


def prompt(
    backend: Backend,
    parser: Union[str, Any] = "str",
    template_file: Optional[str] = None,
    template: Optional[str] = None,
    stop_template: Optional[str] = None,
    stream: bool = False,
    block_input=None,
    block_output=None,
) -> Callable[[Any], Prompt[Input, Output, FnOutput]]:
    return lambda fn: Prompt(
        backend,
        parser,
        template_file,
        template,
        stop_template,
        fn,
        stream,
        block_input,
        block_output,
    )
