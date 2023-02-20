from jinja2 import Template
from .base import Prompt, Request, Output
from typing import Mapping, Any

class SimplePrompt(Prompt[str, str]):
    """
    A simple prompt that echos the string it is passed and returns the 
    output of the LLM as a string.
    """
    @classmethod
    def prompt(cls, inp: str) -> Request:
        return Request(inp)

    @classmethod
    def parse(cls, input: str) -> str:
        return input


class JinjaPrompt(Prompt[Mapping[str, Any], Output]):
    """
    A prompt that uses Jinja2 to define a prompt based on a static template.
    """

    prompt_template = ""
    stop_template = ""

    
    @classmethod
    def prompt(cls, kwargs: Mapping[str, Any]) -> Request:
        x: str = Template(cls.prompt_template).render(**kwargs)
        stop: str = Template(cls.stop_template).render(**kwargs)
        return Request(x, [stop])

