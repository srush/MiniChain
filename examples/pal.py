desc = """
### Prompt-aided Language Models

Chain for answering complex problems by code generation and execution. [[Code](https://github.com/srush/MiniChain/blob/main/examples/pal.py)]

(Adapted from Prompt-aided Language Models [PAL](https://arxiv.org/pdf/2211.10435.pdf)).
"""

# $

from minichain import prompt, show, OpenAI, Python

@prompt(OpenAI(), template_file="pal.pmpt.tpl")
def pal_prompt(model, question):
    return model(dict(question=question))

@prompt(Python())
def python(model, inp):
    return float(model(inp + "\nprint(solution())"))

def pal(question):
    return python(pal_prompt(question))

# $

question = "Melanie is a door-to-door saleswoman. She sold a third of her " \
    "vacuum cleaners at the green house, 2 more to the red house, and half of " \
    "what was left at the orange house. If Melanie has 5 vacuum cleaners left, " \
    "how many did she start with?"

gradio = show(pal,
              examples=[question],
              subprompts=[pal_prompt, python],
              description=desc,
              out_type="json",
              code=open("pal.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )

if __name__ == "__main__":
    gradio.launch()
