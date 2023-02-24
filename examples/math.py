from typing import List
from minichain import Backend, JinjaPrompt, Prompt, start_chain, SimplePrompt


# Prompt that asks LLM to produce a bash command.

class MathPrompt(JinjaPrompt[str]):
    template_file = "math.pmpt.tpl"


with start_chain("math") as backend:
    question = 'What is the sum of the powers of 3 (3^i) that are smaller than 100?'
    prompt = MathPrompt(backend.OpenAI()).chain(SimplePrompt(backend.Python()))
    result = prompt({"question": question})
    print(result)

    
