# + tags=["hide_inp"]
desc = """
### Chat

A chat-like example for multi-turn chat with state. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/chat.ipynb)

(Adapted from [LangChain](https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html)'s version of this [blog post](https://www.engraved.blog/building-a-virtual-machine-inside/).)

"""
# -


# $

from dataclasses import dataclass, replace
from typing import List, Tuple
from minichain import OpenAI, prompt, show

# Generic stateful Memory

MEMORY = 2

@dataclass
class State:
    memory: List[Tuple[str, str]]
    human_input: str = ""

    def push(self, response: str) -> "State":
        memory = self.memory if len(self.memory) < MEMORY else self.memory[1:]
        return State(memory + [(self.human_input, response)])

# Chat prompt with memory

@prompt(OpenAI(), template_file="chat.pmpt.tpl")
def chat_prompt(model, state: State) -> State:
    out = model(state)
    result = out.split("Assistant:")[-1]
    return state.push(result)

# $

examples = [
    "ls ~",
    "cd ~",
    "{Please make a file jokes.txt inside and put some jokes inside}",
    """echo -e "x=lambda y:y*5+3;print('Result:' + str(x(6)))" > run.py && python3 run.py""",
    """echo -e "print(list(filter(lambda x: all(x%d for d in range(2,x)),range(2,3**10)))[:10])" > run.py && python3 run.py""",
    """echo -e "echo 'Hello from Docker" > entrypoint.sh && echo -e "FROM ubuntu:20.04\nCOPY entrypoint.sh entrypoint.sh\nENTRYPOINT [\"/bin/sh\",\"entrypoint.sh\"]">Dockerfile && docker build . -t my_docker_image && docker run -t my_docker_image""",
    "nvidia-smi"
]

gradio = show(lambda command, state: chat_prompt(replace(state, human_input=command)),
              initial_state=State([]),
              subprompts=[chat_prompt],
              examples=examples,
              out_type="json",
              description=desc,
              code=open("chat.py", "r").read().split("$")[1].strip().strip("#").strip(),
)
if __name__ == "__main__":
    gradio.launch()


