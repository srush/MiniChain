import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Type, get_args, get_origin

import numpy as np
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)

from .backend import Request
from .base import HTML, Input, Output, Prompt

env = Environment(
    loader=PackageLoader("minichain"),
    autoescape=select_autoescape(),
    extensions=["jinja2_highlight.HighlightExtension"],
)


class SimplePrompt(Prompt[str, str]):
    """
    A simple prompt that echos the string it is passed and returns the
    output of the LLM as a string.
    """

    def prompt(self, inp: str) -> Request:
        return Request(inp)

    def parse(self, out: str, inp: str) -> str:
        return out


class TemplatePrompt(Prompt[Mapping[str, Any], Output]):
    """
    A prompt that uses Jinja to define a prompt based on a static template.
    Set `template_file` to the Jinja template file.
    """

    IN = Mapping[str, Any]
    template = None
    template_file = ""
    prompt_template = ""
    stop_template = ""

    def to_dict(self, inp):
        return inp
        
    
    def render_prompt_html(self, inp: Mapping[str, Any], prompt: str) -> HTML:
        n = {}
        if not isinstance(inp, dict):
            inp = asdict(inp)

        for k, v in inp.items():
            if isinstance(v, str):
                n[k] = f"<div style='color:red'>{v}</div>"
            else:
                n[k] = v
        return HTML(self.prompt(n).prompt.replace("\n", "<br>"))

    def parse(self, result: str, inp: Mapping[str, Any]) -> Output:
        return str(result)  # type: ignore

    def prompt(self, inp: Input) -> Request:
        kwargs = self.to_dict(inp)
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = self.template  # type: ignore
        else:
            tmp = Template(self.prompt_template)
        if isinstance(kwargs, dict):
            x = tmp.render(**kwargs)
        else:
            x = tmp.render(**asdict(kwargs))

        if self.stop_template:
            stop = [Template(self.stop_template).render(**kwargs)]
        else:
            stop = None
        return Request(x, stop)


class EmbeddingPrompt(Prompt[Input, Output]):
    """
    A prompt that replaces parse with `find` that takes an embedding.
    """

    def parse(self, response: str, inp: Input) -> Output:
        return self.find(response, inp)

    def find(self, response: np.ndarray, inp: Input) -> Output:
        """
        Convert from the embedding response of the function
        to the output type.
        """
        raise NotImplementedError


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


class TypedTemplatePrompt(TemplatePrompt[Output]):
    """
    Prompt that is automatically generated to produce a
    list of objects of of the dataclass `Out`.
    """

    Out = None

    def prompt(self, inp: TemplatePrompt.IN) -> Request:
        inp = dict(inp)
        tmp = env.get_template("type_prompt.pmpt.tpl")
        d = walk(self.Out)
        inp["typ"] = tmp.render({"typ": d})

        return super().prompt(inp)

    def parse(self, out: str, inp: TemplatePrompt.IN) -> Output:
        return [self.Out(**j) for j in json.loads(out)]  # type: ignore
