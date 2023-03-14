
desc = """
### Typed Extraction

Information extraction that is automatically generated from a typed specification. [[Code](https://github.com/srush/MiniChain/blob/main/examples/stats.py)]

(Novel to MiniChain)
"""

# $

import minichain
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


# Code

class ExtractionPrompt(minichain.TypedTemplatePrompt):
    template_file = "stats.pmpt.tpl"
    Out = Player


with minichain.start_chain("stats") as backend:
    prompt = ExtractionPrompt(backend.OpenAI(max_tokens=512))

# $


article = open("sixers.txt").read()
gradio = prompt.to_gradio(fields =["passage"],
                          examples=[article],
                          out_type="json",
                          description=desc,
                          code=open("stats.py", "r").read().split("$")[1].strip().strip("#").strip(),
                          templates=[open("stats.pmpt.tpl")]
)
if __name__ == "__main__":
    gradio.launch()

    
# ExtractionPrompt().show({"passage": "Harden had 10 rebounds."},
#                         '[{"player": "Harden", "stats": {"value": 10, "stat": 2}}]')

# # View the run log.

# minichain.show_log("bash.log")
