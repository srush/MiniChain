# + tags=["hide_inp"]
desc = """
### Word Problem Solver

Chain that solves a math word problem by first generating and then running Python code. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/math_demo.ipynb)

(Adapted from Dust [maths-generate-code](https://dust.tt/spolu/a/d12ac33169))
"""
# -

# $

from minichain import show, prompt, OpenAI, Python, GradioConf
import gradio as gr


@prompt(OpenAI(), template_file="math.pmpt.tpl",
        gradio_conf=GradioConf(block_input=gr.Markdown))
def math_prompt(model, question
                ):
    "Prompt to call GPT with a Jinja template"
    return model(dict(question=question))

@prompt(Python(), template="import math\n{{code}}")
def python(model, code):
    "Prompt to call Python interpreter"
    code = "\n".join(code.strip().split("\n")[1:-1])
    return model(dict(code=code))

def math_demo(question):
    "Chain them together"
    return python(math_prompt(question))

# $

# + tags=["hide_inp"]
gradio = show(math_demo,
              examples=["What is the sum of the powers of 3 (3^i) that are smaller than 100?",
                        "What is the sum of the 10 first positive integers?",],
                        # "Carla is downloading a 200 GB file. She can download 2 GB/minute, but 40% of the way through the download, the download fails. Then Carla has to restart the download from the beginning. How load did it take her to download the file in minutes?"],
              subprompts=[math_prompt, python],
              description=desc,
              code=open("math_demo.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )
if __name__ == "__main__":
    gradio.queue().launch()
# -

