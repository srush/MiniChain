from typing import Protocol, Any, TypeVar, Generic, Mapping, List, Sequence
from eliot import start_action, to_file
from dataclasses import dataclass
from .backend import Backend


Input = TypeVar("Input")
Output = TypeVar("Output")


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
    raw: str

    

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
    
    @classmethod
    def prompt(cls, inp: Input) -> Request:
        raise NotImplementedError

    @classmethod
    def parse(cls, out: str) -> Output:
        raise NotImplementedError

    @classmethod
    def run(cls, backend: Backend, inp: Input, name: str="") -> Response[Output]:
        with start_action(action_type=str(cls) + name):
            with start_action(action_type="Input Function", input=inp):
                request : Request = cls.prompt(inp)
            with start_action(action_type="Prompted", prompt=request.prompt):
                result: str = backend.run(request.prompt, request.stop)
            with start_action(action_type="Result", result=result):
                output: Output = cls.parse(result)
                start_action(action_type="Return", returned=output)
                
        return Response(output, request.prompt + result, result)

    @classmethod
    async def arun(cls, backend: Backend, inp: Input, name: str="") -> Response[Output]:
        return cls.run(backend, inp, name=name)


