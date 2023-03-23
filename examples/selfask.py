
desc = """
### Self-Ask

 Notebook implementation of the self-ask + Google tool use prompt. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/selfask.ipynb)

 (Adapted from [Self-Ask repo](https://github.com/ofirpress/self-ask))
"""

# $

from dataclasses import dataclass, replace
from typing import Optional
from minichain import prompt, show, OpenAI, Google


@dataclass
class State:
    question: str
    history: str =  ""
    next_query: Optional[str] = None
    final_answer: Optional[str] = None


@prompt(OpenAI(),
        template_file = "selfask.pmpt.tpl",
        stop_template = "\nIntermediate answer:")
def self_ask(model, state):
    out = model(state)
    res = out.split(":", 1)[1]
    if out.startswith("Follow up:"):
        return replace(state, next_query=res)
    elif out.startswith("So the final answer is:"):
        return replace(state, final_answer=res)

@prompt(Google())
def google(model, state):
    if state.next_query is None:
        return state

    result = model(state.next_query)
    return State(state.question,
                 state.history + "\nIntermediate answer: " + result + "\n")

def selfask(question):
    state = State(question)
    for i in range(3):
        state = self_ask(state)
        state = google(state)
    return state

# $

gradio = show(selfask,
              examples=["What is the zip code of the city where George Washington was born?"],
              subprompts=[self_ask, google] * 3,
              description=desc,
              code=open("selfask.py", "r").read().split("$")[1].strip().strip("#").strip(),
              out_type="json"
              )
if __name__ == "__main__":
    gradio.launch()


