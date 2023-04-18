import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from itertools import count
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import gradio as gr
from eliot import start_action
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)

from .backend import Backend, MinichainContext



Input = TypeVar("Input")
Output = TypeVar("Output")
FnOutput = TypeVar("FnOutput")


@dataclass
class History:
    expand: Callable[[List[Any]], Iterator[Any]]
    inputs: List[Any]


@dataclass
class Chain:
    history: History

    def run_gen(self) -> Any:
        # Lazily instantiate all the inputs
        args = []
        for base_input in self.history.inputs:
            function_input = base_input
            if isinstance(base_input, Chain):
                for function_input in base_input.run_gen():
                    yield None
            args.append(function_input)
            
        # Run the current prompt
        for out in self.history.expand(*args):
            yield None
        yield out

    def run(self) -> Any:
        for x in self.run_gen():
            pass
        return x

            
@dataclass
class RunLog:
    request: str = ""
    response: str = ""
    output: str = ""
    dynamic: int = 0


@dataclass
class PromptSnap:
    input_: Any = ""
    run_log: RunLog = RunLog()
    output: str = ""


class Prompt(Generic[Input, Output, FnOutput]):
    counter = count()

    def __init__(
        self,
        fn: Callable[[Callable[[Input], Output]], FnOutput],
        backend: Optional[Backend],
        template_file: Optional[str],
        template: Optional[str],
        gradio_conf = None,

    ):
        self.fn = fn
        if not isinstance(backend, List):
            self.backend: Optional[Backend] = [backend]
        else:
            self.backend = backend

        self.template_file: Optional[str] = template_file
        self.template: Optional[str] = template
        self.gradio_conf = gradio_conf
            
        self._fn: str = fn.__name__
        self._id: int = Prompt.counter.__next__()

    def run(self, request: str, tool_num=0) -> Iterator[RunLog]:
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

        return tmp.render(**kwargs)

    def __call__(self, *args: Any) -> FnOutput:
        return Chain(History(self.expand, args))

    class Model:
        def __init__(self, prompt: "Prompt", data: Any):
            self.prompt = prompt
            self.data = data
            self.run_log = RunLog()

        def __call__(self, model_input: Any, tool_num: int=0) -> Any:          
            for r in self.stream(model_input, tool_num):             
                yield r



            # print("hello tool")                   
            # for out in self.prompt.dynamic[tool_num].expand(*model_input):
            #     self.run_log = self.prompt.dynamic[tool_num].model.run_log
            #     self.run_log.dynamic = tool_num
            #     yield out

        def stream(self, model_input: Any, tool_num:int=0) -> Iterator[str]:
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
                self.run_log = RunLog(
                    run_log.request,
                    out,
                    dynamic=tool_num
                )
                yield self.run_log.response
                
    def expand(
        self, *args: List[Any], data: Any = None
    ) -> Iterator[str]:
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
    backend: Optional[Backend] = None,
    template_file: Optional[str] = None,
    template: Optional[str] = None,
    gradio_conf = None
) -> Callable[[Any], Prompt[Input, Output, FnOutput]]:
    return lambda fn: Prompt(
        fn,
        backend,
        template_file,
        template,
        gradio_conf
    )

def transform():
    return lambda fn: lambda *args: Chain(History(lambda *x: (fn(*x),), args))
