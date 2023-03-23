import json
from dataclasses import asdict, dataclass, is_dataclass, fields
from itertools import count
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union, Type, get_args, get_origin, Dict
from enum import Enum
from eliot import start_action
from jinja2 import Environment, FileSystemLoader, Template, PackageLoader,  select_autoescape


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

def type_to_prompt(Out) -> str:
    tmp = env.get_template("type_prompt.pmpt.tpl")
    d = walk(Out)
    return tmp.render({"typ": d})

def simple(model, **kwargs):  # type: ignore
    return model(kwargs)

@dataclass
class History:
    prompt: "Prompt"
    inputs: List[Any]

@dataclass
class Fail:
    arg_num: int
    data: Any
    
@dataclass
class Chain:
    # TODO: Add caching
    history: History
    
    def run(self, trial=0, data=None) -> Any:
        args = []
        count = []
        for inp in self.history.inputs:
            if isinstance(inp, Chain):
                inp = inp.run()
            args.append(inp)
            count.append(0)
        out = self.history.prompt.expand(args, trial, data)
        while isinstance(out, Fail):
            count[out.arg_num] += 1
            inp = self.history.inputs[out.arg_num]
            assert isinstance(inp, Chain)
            args[out.arg_num] = inp.run(trial=count[out.arg_num],
                                        data=out.data)
            out = self.history.prompt.expand(args, trial, data)
            
        return out 
    
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
        return Chain(History(self, args))

    class Model:
        def __init__(self, prompt, trial, data):
            self.prompt = prompt
            self.trial = trial
            self.data = data
            self.run_log = None

        def fail(self, argnum, data=None):
            return Fail(argnum-1, data)
            
        def __call__(self, input_):
            assert self.run_log is None, "Only call `model` once per function"
            if self.prompt.template is not None or self.prompt.template_file is not None:
                input_ = dict(**input_)
                input_["_trial"] = self.trial
                input_["_fail_data"] = self.data

                result = self.prompt.template_fill(input_)
            else:
                result = input_
            self.run_log = self.prompt.run_verbose(result)
            return self.run_log[-1]            
    
    def expand(self, args, trial=0, data=None):
        model = self.Model(self, trial, data)
        output = self.fn(model, *args)
        if not isinstance(output, Fail):
            t = model.run_log
            MinichainContext.prompt_count.setdefault(self._id, -1)
            if trial == 0:
                MinichainContext.prompt_count[self._id] += 1
            count = MinichainContext.prompt_count[self._id]
            MinichainContext.prompt_store.setdefault((self._id, count), [])
            MinichainContext.prompt_store[self._id, count].append((args, t[0], t[1], output))
        return output
    
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
