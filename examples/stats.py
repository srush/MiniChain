
desc = """
### Typed Extraction

Information extraction that is automatically generated from a typed specification. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/pal.ipynb)

(Novel to MiniChain)
"""

# $

from minichain import prompt, show, type_to_prompt, OpenAI
from dataclasses import dataclass
from typing import List
from enum import Enum

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


@prompt(OpenAI(), template_file="stats.pmpt.tpl", parser="json")
def stats(model, passage):
    out = model(dict(passage=passage, typ=type_to_prompt(Player)))
    return [Player(**j) for j in out]  

# $

article = open("sixers.txt").read()
gradio = show(lambda passage: stats(passage),
              examples=[article],
              subprompts=[stats],
              out_type="json",
              description=desc,
              code=open("stats.py", "r").read().split("$")[1].strip().strip("#").strip(),
)
if __name__ == "__main__":
    gradio.launch()


# ExtractionPrompt().show({"passage": "Harden had 10 rebounds."},
#                         '[{"player": "Harden", "stats": {"value": 10, "stat": 2}}]')

# # View the run log.

# minichain.show_log("bash.log")
