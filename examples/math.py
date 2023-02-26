# Answer a math problem with code.
# Adapted from Dust [maths-generate-code](https://dust.tt/spolu/a/d12ac33169)

from minichain import SimplePrompt, TemplatePrompt, show_log, start_chain

# Prompt that asks LLM for code from math.


class MathPrompt(TemplatePrompt[str]):
    template_file = "math.pmpt.tpl"


with start_chain("math") as backend:
    question = "What is the sum of the powers of 3 (3^i) that are smaller than 100?"
    prompt = MathPrompt(backend.OpenAI()).chain(SimplePrompt(backend.Python()))
    result = prompt({"question": question})
    print(result)

# + tags=["hide_inp"]
MathPrompt().show({"question": "What is 10 + 12?"}, "10 + 12")
# -

show_log("math.log")
