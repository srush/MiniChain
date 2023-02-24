# Prompt from ...
#
from dataclasses import dataclass

from parsita import TextParsers, lit, reg

from minichain import Backend, JinjaPrompt, Prompt, start_chain


class PalPrompt(JinjaPrompt[str]):
    template_file = "pal.pmpt.tpl"

class PyPrompt(Prompt[str, int]):
    def prompt(self, inp):
        return inp + "\nprint(solution())"

    def parse(self, response, inp):
        return int(response)
    
    
with start_chain("pal") as backend:
    question = 'Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?'
    prompt = PalPrompt(backend.OpenAI()).chain(PyPrompt(backend.Python()))
    result = prompt({"question": question})
    print(result)


    

# !eliot-tree -l 0 selfask.log | grep -v "succeeded" | grep -v "started"
    
