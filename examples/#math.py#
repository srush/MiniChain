# Notebook to answer a math problem with code.
# Adapted from Dust [maths-generate-code](https://dust.tt/spolu/a/d12ac33169)

import minichain

# Prompt that asks LLM for code from math.

class MathPrompt(minichain.TemplatePrompt[str]):
    template_file = "math.pmpt.tpl"

# Ask a question and run it as python code.

with minichain.start_chain("math") as backend:
    question = "What is the sum of the powers of 3 (3^i) that are smaller than 100?"
    prompt = MathPrompt(backend.OpenAI()).chain(minichain.SimplePrompt(backend.Python()))
    result = prompt({"question": question})
    print(result)

# View the prompt
    
# + tags=["hide_inp"]
MathPrompt().show({"question": "What is 10 + 12?"}, "10 + 12")
# -

# View the log

minichain.show_log("math.log")
