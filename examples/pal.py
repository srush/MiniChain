# Adapted from Prompt-aided Language Models [PAL](https://arxiv.org/pdf/2211.10435.pdf).

from minichain import Backend, JinjaPrompt, Prompt, start_chain, show_log

class PalPrompt(JinjaPrompt[str]):
    template_file = "pal.pmpt.tpl"
PalPrompt().show({"question": "Joe has 10 cars and Bobby has 12 cars. How many do they have together?"},
                 "def solution():\n\treturn 10 + 12")

class PyPrompt(Prompt[str, int]):
    def prompt(self, inp):
        return inp + "\nprint(solution())"

    def parse(self, response, inp):
        return int(response)
PyPrompt().show("def solution():\n\treturn 10 + 12", "22")
    
with start_chain("pal") as backend:
    question = 'Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?'
    prompt = PalPrompt(backend.OpenAI()).chain(PyPrompt(backend.Python()))
    result = prompt({"question": question})
    print(result)


show_log("pal.log")    
