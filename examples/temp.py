import minichain
from dataclasses import fields, dataclass, is_dataclass
from typing import List
from enum import Enum

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

class T(minichain.TypedTemplatePrompt):
    template_file = "tmp.tpl"
    Output = Player

print(T().show({"passage": "hello"}, '[{"player": "Harden", "stats": {"value": 10, "stat": 2}}]'))

with minichain.start_chain("stats") as backend:
    p = T(backend.OpenAI(max_tokens=512))
    print(p({"passage": open("sixers.txt").read()}))

# def enum(x):
#     d = {e.name: e.value for e in x}
#     # d["__enum__"] = True
#     return d
    
            
# def walk(x):
#     print(x)
#     if issubclass(x, Enum):
#         return enum(x)
#     if is_dataclass(x):
#         return {y.name: walk(y.type) for y in fields(x)}
#     return x.__name__
#     # return [x for x in fields(B)]
#             # print(x.name)
#             # print(x.type)
#             # if issubclass(x.type, Enum):
#             # for e in x.type:
#             # print(e.value)
#             # print(e.name)
#             # print(x)]

# print(walk(B))
