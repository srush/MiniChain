from dataclasses import dataclass
from typing import Generic, Sequence, Tuple, TypeVar

from eliot import start_action, to_file

from .backend import Backend

Input = TypeVar("Input")
Input2 = TypeVar("Input2")
Output = TypeVar("Output")
Output2 = TypeVar("Output2")


@dataclass
class Request:
    prompt: str
    stop: Sequence[str] = ()


@dataclass
class Response(Generic[Output]):
    """
    The return value of calling a prompt.
    """

    val: Output
    echo: str


class Prompt(Generic[Input, Output]):
    """
    `Prompt` represents a typed function from Input to Output.
    It computes its output value by calling an external services, generally
    a large language model (LLM).

    To define a new prompt, you need to inherit from this class and
    define the function `prompt` which takes a argument of type
    `Input` and produces a string to be sent to the LLM, and the
    function `parse` which takes a string and produces a value of type
    `Output`.


    To use the `Prompt` you call `run` or `arun` with an LLM backend and the arguments
    of the form `Input`.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

    def prompt(self, inp: Input) -> Request | str:
        raise NotImplementedError

    def parse(self, out: str) -> Output:
        raise NotImplementedError

    def __call__(self, inp: Input, name: str = "") -> Response[Output]:
        with start_action(action_type=str(type(self)) + name):
            with start_action(action_type="Input Function", input=inp):
                r = self.prompt(inp)
            if isinstance(r, str):
                request = Request(r)
            else:
                request = r

            with start_action(action_type="Prompted", prompt=request.prompt):
                result: str = self.backend.run(request.prompt, request.stop)
            with start_action(action_type="Result", result=result):
                output: Output = self.parse(result)
                start_action(action_type="Return", returned=output)

        return Response(output, request.prompt + result)

    def chain(self, other: "Prompt[Output, Output2]") -> "Prompt[Input, Output2]":
        return ChainedPrompt(self, other)

    def par(
        self, other: "Prompt[Input2, Output2]"
    ) -> "Prompt[Tuple[Input, Input2], Tuple[Output, Output2]]":
        return ParallelPrompt(self, other)

    def map(self) -> "Prompt[Sequence[Input], Sequence[Output]]":
        return MapPrompt(self)

    async def arun(self, inp: Input, name: str = "") -> Response[Output]:
        return self(inp, name=name)


class ChainedPrompt(Prompt[Input, Output2]):
    def __init__(
        self, prompt1: Prompt[Input, Output], prompt2: Prompt[Output, Output2]
    ):
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def __call__(self, inp: Input, name: str = "") -> Response[Output2]:
        out = self.prompt1(inp)
        out2 = self.prompt2(out.val)
        return Response(out2.val, out.echo + out2.echo)


class ParallelPrompt(Prompt[Tuple[Input, Input2], Tuple[Output, Output2]]):
    def __init__(
        self, prompt1: Prompt[Input, Output], prompt2: Prompt[Input2, Output2]
    ):
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def __call__(
        self, inp: Tuple[Input, Input2], name: str = ""
    ) -> Response[Tuple[Output, Output2]]:
        out1 = self.prompt1(inp[0])
        out2 = self.prompt2(inp[1])
        return Response((out1.val, out2.val), out1.echo + out2.echo)


class MapPrompt(Prompt[Sequence[Input], Sequence[Output]]):
    def __init__(self, prompt: Prompt[Input, Output]):
        self.prompt1 = prompt

    def __call__(
        self, inp: Sequence[Input], name: str = ""
    ) -> Response[Sequence[Output]]:
        vals = []
        echo = ""
        for i in inp:
            out = self.prompt1(i)
            vals.append(out.val)
            echo += out.echo
        return Response(vals, echo)

    # def __call__(self, inp: Input):
    #     async with trio.open_nursery() as nursery:
    #         nursery.start_soon(mock.ask, SimplePrompt, dict(input="b", name=f"F1"))
    #         nursery.start_soon(mock.ask, SimplePrompt, dict(input="a", name=f"F2"))

    # out = self.prompt1(inp)
    # return self.prompt2(out)
