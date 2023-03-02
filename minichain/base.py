from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, TypeVar, Union, Any

import trio
from eliot import start_action
from jinja2 import Environment, PackageLoader, select_autoescape

from .backend import Backend, Request

# Type Variables.

Input = TypeVar("Input")
Input2 = TypeVar("Input2")
Output = TypeVar("Output")
Output2 = TypeVar("Output2")


# Code for visualization.
env = Environment(
    loader=PackageLoader("minichain"),
    autoescape=select_autoescape(),
    extensions=["jinja2_highlight.HighlightExtension"],
)


@dataclass
class HTML:
    html: str

    def _repr_html_(self) -> str:
        return self.html


# Main Class


class Prompt(Generic[Input, Output]):
    """`Prompt` represents a typed function from Input to Output.
    It computes its output value by calling an external services, generally
    a large language model (LLM).

    To define a new prompt, you need to inherit from this class and
    define the function `prompt` which takes a argument of type
    `Input` and produces a string to be sent to the LLM, and the
    function `parse` which takes a string and produces a value of type
    `Output`.


    To use the `Prompt` you call `__call__` or `arun` with an LLM
    backend and the arguments of the form `Input`.

    """

    def __init__(self, backend: Optional[Backend] = None, data: Any = None):
        self.backend = backend
        self.data = data
        
    # START: Overloaded by the user prompts
    def prompt(self, inp: Input) -> Union[Request, str]:
        """
        Convert from the `Input` type of the function
        to a request that is sent to the backend.
        """
        return str(inp)

    def parse(self, response: str, inp: Input) -> Output:
        """
        Convert from the string response of the function
        to the output type.
        """
        raise NotImplementedError

    # END: Overloaded by the user prompts

    def _prompt(self, inp: Input) -> Request:
        with start_action(action_type="Input Function", input=inp):
            r = self.prompt(inp)
            if isinstance(r, str):
                return Request(r)
            else:
                return r

    def __call__(self, inp: Input) -> Output:
        assert self.backend is not None
        with start_action(action_type=str(type(self))):
            request = self._prompt(inp)
            with start_action(action_type="Prompted", prompt=request.prompt):
                result: str = self.backend.run(request)
            with start_action(action_type="Result", result=result):
                output = self.parse(result, inp)
        return output

    async def arun(self, inp: Input) -> Output:
        assert self.backend is not None
        with start_action(action_type=str(type(self))):
            request = self._prompt(inp)
            with start_action(action_type="Prompted", prompt=request.prompt):
                result = await self.backend.arun(request)
            with start_action(action_type="Result", result=result):
                output = self.parse(result, inp)
        return output

    def render_prompt_html(self, inp: Input, prompt: str) -> HTML:
        return HTML(prompt.replace("\n", "<br>"))

    # Other functions.

    def show(self, inp: Input, response: str) -> HTML:
        prompt = self.prompt(inp)
        if isinstance(prompt, Request):
            prompt = prompt.prompt
        tmp = env.get_template("prompt.html.tpl")
        return HTML(
            tmp.render(
                **{
                    "name": type(self).__name__,
                    "input": str(inp),
                    "response": response,
                    "output": str(self.parse(response, inp)),
                    "prompt": self.render_prompt_html(inp, prompt).html,
                }
            )
        )

    def chain(self, other: "Prompt[Output, Output2]") -> "Prompt[Input, Output2]":
        "Chain together two prompts"
        return ChainedPrompt(self, other)

    def map(self) -> "Prompt[Sequence[Input], Sequence[Output]]":
        "Create a prompt the works on lists of inputs"
        return MapPrompt(self)


class ChainedPrompt(Prompt[Input, Output2]):
    def __init__(
        self, prompt1: Prompt[Input, Output], prompt2: Prompt[Output, Output2]
    ):
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def __call__(self, inp: Input) -> Output2:
        out = self.prompt1(inp)
        return self.prompt2(out)

    async def arun(self, inp: Input) -> Output2:
        out = await self.prompt1.arun(inp)
        return await self.prompt2.arun(out)


class MapPrompt(Prompt[Sequence[Input], Sequence[Output]]):
    def __init__(self, prompt: Prompt[Input, Output]):
        self.prompt1 = prompt

    def __call__(self, inp: Sequence[Input]) -> Sequence[Output]:
        return [self.prompt1(i) for i in inp]

    async def arun(self, inp: Sequence[Input]) -> Sequence[Output]:
        results: List[Optional[Output]] = [None] * len(inp)

        async def runner(i: int) -> None:
            results[i] = await self.prompt1.arun(inp[i])

        async with trio.open_nursery() as nursery:
            for i in range(len(inp)):
                nursery.start_soon(runner, i)

        ret: List[Output] = []
        for i in range(len(inp)):
            r = results[i]
            assert r is not None
            ret.append(r)
        return ret
