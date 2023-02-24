from typing import List
from minichain import Backend, JinjaPrompt, Prompt, start_chain
import trio


# Prompt that asks LLM to produce a bash command.

class SummaryPrompt(JinjaPrompt[str]):
    template_file = "summary.pmpt.tpl"

def chunk(f):
    text = open(f).read().replace("\n\n", "\n")
    chunks = []
    W = 4000
    O = 800
    for i in range(4):
        if i* W > len(text):
            break
        chunks.append({"text": text[i * W: (i+1) * W + O]})
    return chunks

with start_chain("summary") as backend:
    prompt = SummaryPrompt(backend.OpenAI())
    list_prompt = prompt.map()
    out = trio.run(list_prompt.arun, chunk("../state_of_the_union.txt"))
    print(prompt({"text": "\n".join(out)}))
     
