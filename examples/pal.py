desc = """
### Prompt-aided Language Models

Chain for answering complex problems by code generation and execution. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/pal.ipynb)

(Adapted from Prompt-aided Language Models [PAL](https://arxiv.org/pdf/2211.10435.pdf)).
"""

# $

from minichain import prompt, show, GradioConf, OpenAI, Python
import gradio as gr

@prompt(OpenAI(), template_file="pal.pmpt.tpl")
def pal_prompt(model, question):
    return model(dict(question=question))

@prompt(Python(),
        gradio_conf=GradioConf(block_input = lambda: gr.Code(language="python")))
def python(model, inp):
    return model(inp + "\nprint(solution())")

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
    gradio.queue().launch()
