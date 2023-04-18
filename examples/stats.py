
desc = """
### Typed Extraction

Information extraction that is automatically generated from a typed specification. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/pal.ipynb)

(Novel to MiniChain)
"""

# $

from minichain import prompt, show, OpenAI, transform
from dataclasses import dataclass, is_dataclass, fields
from typing import List, Type, Dict, Any, get_origin, get_args
from enum import Enum
from jinja2 import select_autoescape, FileSystemLoader, Environment
import json

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

env = Environment(
    loader=FileSystemLoader("."),
    autoescape=select_autoescape(),
    extensions=["jinja2_highlight.HighlightExtension"],
)



# Data specification

# +
class StatType(Enum):
    POINTS = 1
    REBOUNDS = 2
    ASSISTS = 3

@dataclass
class Stat:
    value: int
    stat: StatType

@dataclass
class Player:
    player: str
    stats: List[Stat]
# -


@prompt(OpenAI(), template_file="stats.pmpt.tpl")
def stats(model, passage):
    return model.stream(dict(passage=passage, typ=type_to_prompt(Player)))

@transform()
def to_data(s:str):
    return [Player(**j) for j in json.loads(s)]

# $

article = open("sixers.txt").read()
gradio = show(lambda passage: to_data(stats(passage)),
              examples=[article],
              subprompts=[stats],
              out_type="json",
              description=desc,
              code=open("stats.py", "r").read().split("$")[1].strip().strip("#").strip(),
)
if __name__ == "__main__":
    gradio.queue().launch()


# ExtractionPrompt().show({"passage": "Harden had 10 rebounds."},
#                         '[{"player": "Harden", "stats": {"value": 10, "stat": 2}}]')

# # View the run log.

# minichain.show_log("bash.log")
