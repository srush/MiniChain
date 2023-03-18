# + tags=["hide_inp"]
desc = """
### Chat

"ChatGPT" like example for multi-turn chat with state. [[Code](https://github.com/srush/MiniChain/blob/main/examples/chat.py)]

(Adapted from [LangChain](https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html)'s version of this [blog post](https://www.engraved.blog/building-a-virtual-machine-inside/).)

Recommended use: input examples in order one at a time. 
"""
# -


import warnings

# + tags=["hide_inp"]
warnings.filterwarnings("ignore")
# -

# $

from dataclasses import dataclass
from typing import List, Tuple
import minichain

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

class ChatPrompt(minichain.TemplatePrompt):
    template_file = "chatgpt.pmpt.tpl"
    def parse(self, out: str, inp: State) -> State:
        result = out.split("Assistant:")[-1]
        return inp.push(result)

# class Human(minichain.Prompt):
#     def parse(self, out: str, inp: State) -> State:
#         return inp.human_input = out
    

with minichain.start_chain("chat") as backend:
    prompt = ChatPrompt(backend.OpenAI())
    state = State([])

# $
    
examples = [
    "I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is pwd.",
    "ls ~",
    "cd ~",
    "{Please make a file jokes.txt inside and put some jokes inside}",
    """echo -e "x=lambda y:y*5+3;print('Result:' + str(x(6)))" > run.py && python3 run.py""",
    """echo -e "print(list(filter(lambda x: all(x%d for d in range(2,x)),range(2,3**10)))[:10])" > run.py && python3 run.py""",
    """echo -e "echo 'Hello from Docker" > entrypoint.sh && echo -e "FROM ubuntu:20.04\nCOPY entrypoint.sh entrypoint.sh\nENTRYPOINT [\"/bin/sh\",\"entrypoint.sh\"]">Dockerfile && docker build . -t my_docker_image && docker run -t my_docker_image""",
    "nvidia-smi"
]

gradio = prompt.to_gradio(fields= ["human_input"],
                          initial_state= state,
                          examples=examples,
                          out_type="json",
                          description=desc,
                          code=open("chat.py", "r").read().split("$")[1].strip().strip("#").strip(),
                          templates=[open("chat.pmpt.tpl")]
                                
)
if __name__ == "__main__":
    gradio.launch()

# for i in range(len(fake_human)):
#     human.chain(prompt)

    
