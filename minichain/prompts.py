from typing import Any, Mapping

from jinja2 import Template

from .base import Output, Prompt, Request


class SimplePrompt(Prompt[str, str]):
    """
    A simple prompt that echos the string it is passed and returns the
    output of the LLM as a string.
    """

    def prompt(self, inp: str) -> Request:
        return Request(inp)

    def parse(self, input: str) -> str:
        return input


class JinjaPrompt(Prompt[Mapping[str, Any], Output]):
    """
    A prompt that uses Jinja2 to define a prompt based on a static template.
    """

    template_file = ""
    prompt_template = ""
    stop_template = ""

    def prompt(self, kwargs: Mapping[str, Any]) -> Request:
        x: str = Template(self.prompt_template).render(**kwargs)
        stop: str = Template(self.stop_template).render(**kwargs)
        return Request(x, [stop])
