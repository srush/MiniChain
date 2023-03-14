desc = """
### Prompt-aided Language Models

Chain for answering complex problems by code generation and execution. [[Code](https://github.com/srush/MiniChain/blob/main/examples/pal.py)]

(Adapted from Prompt-aided Language Models [PAL](https://arxiv.org/pdf/2211.10435.pdf)).
"""

# $

import minichain

# PAL Prompt

class PalPrompt(minichain.TemplatePrompt):
    template_file = "pal.pmpt.tpl"

# Prompt to run and print python code.

class PyPrompt(minichain.Prompt):
    def prompt(self, inp):
        return inp + "\nprint(solution())"

    def parse(self, response, inp):
        return int(response)

# Chain the prompts.

with minichain.start_chain("pal") as backend:
    prompt = PalPrompt(backend.OpenAI()).chain(PyPrompt(backend.Python()))
    
# $
    
question = "Melanie is a door-to-door saleswoman. She sold a third of her " \
    "vacuum cleaners at the green house, 2 more to the red house, and half of " \
    "what was left at the orange house. If Melanie has 5 vacuum cleaners left, " \
    "how many did she start with?"
    
gradio = prompt.to_gradio(fields =["question"],
                          examples=[question],
                          description=desc,
                          code=open("pal.py", "r").read().split("$")[1].strip().strip("#").strip(),
                          templates=[open("pal.pmpt.tpl")]
                          )
if __name__ == "__main__":
    gradio.launch()

# View prompt examples.

# # + tags=["hide_inp"]
# PalPrompt().show(
#     {"question": "Joe has 10 cars and Bobby has 12. How many do they have together?"},
#     "def solution():\n\treturn 10 + 12",
# )
# # -

# # + tags=["hide_inp"]
# PyPrompt().show("def solution():\n\treturn 10 + 12", "22")
# # -

# # View the log.

# minichain.show_log("pal.log")
