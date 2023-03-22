import json
from dataclasses import asdict, dataclass
from itertools import count
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union, Type, get_args, get_origin, Dict
from enum import Enum
from eliot import start_action
from jinja2 import Environment, FileSystemLoader, Template

from .backend import Backend, MinichainContext, Request


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

def type_to_prompt(Out) -> str:
    inp = dict(inp)
    tmp = env.get_template("type_prompt.pmpt.tpl")
    d = walk(Out)
    return tmp.render({"typ": d})

def simple(model, **kwargs):  # type: ignore
    return model(kwargs)

@dataclass
class Chain:
    data: Any

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
    ):
        self.backend: Backend = backend
        self.parser: Union[str, Callable[[str], Output]] = parser
        self.template_file: Optional[str] = template_file
        self.template: Optional[str] = template
        self.stop_template: Optional[str] = stop_template
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

    def run_verbose(self, r: Union[str, Request]) -> Tuple[Request, str, Output]:
        # assert self.backend is not None
        with start_action(action_type=str(self.fn)):
            request = _prompt(r)
            with start_action(action_type="Prompted", prompt=request.prompt):
                response: Union[str, Any] = self.backend.run(request)
            with start_action(action_type="Response", result=response):
                output = self.parse(response)
        if not isinstance(response, str):
            response = "(data)"
        return (request, response, output)

    def template_fill(self, inp: Any) -> Request:
        kwargs = inp
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = Template(self.template)

        if not isinstance(kwargs, dict):
            kwargs = asdict(kwargs)
        x = tmp.render(**kwargs)

        if self.stop_template:
            stop = [Template(self.stop_template).render(**kwargs)]
        else:
            stop = None
        return Request(x, stop)

    def __call__(self, *args: Any) -> FnOutput:
        verbose: List[Tuple[Request, str, Output]] = []

        def model(input_: Any) -> Output:
            assert len(verbose) == 0, "Only call `model` once per function"

            if self.template is not None or self.template_file is not None:
                result = self.template_fill(input_)
            else:
                result = input_
            verbose.append(self.run_verbose(result))
            return verbose[0][-1]

        def unwrap(a):
            if isinstance(a, Chain):
                return a.data
            else:
                return a
        args = [unwrap(a) for a in args]
        output = self.fn(model, *args)
        t = verbose[0]
        MinichainContext.prompt_count.setdefault(self._id, 0)
        count = MinichainContext.prompt_count[self._id]
        MinichainContext.prompt_store[self._id, count] = (args, t[0], t[1], output)
        MinichainContext.prompt_count[self._id] += 1
        return Chain(output)


def prompt(
    backend: Backend,
    parser: Union[str, Any] = "str",
    template_file: Optional[str] = None,
    template: Optional[str] = None,
    stop_template: Optional[str] = None,
) -> Callable[[Any], Prompt[Input, Output, FnOutput]]:
    return lambda fn: Prompt(
        backend, parser, template_file, template, stop_template, fn
    )
