# Generate and run a bash command.
# Adapted from LangChain
# [BashChain](https://langchain.readthedocs.io/en/latest/modules/chains/examples/llm_bash.html)

from typing import List

from minichain import TemplatePrompt, Prompt, show_log, start_chain

# Prompt that asks LLM to produce a bash command.


class CLIPrompt(TemplatePrompt[List[str]]):
    template_file = "bash.pmpt.tpl"

    def parse(self, out: str, inp: TemplatePrompt.IN) -> List[str]:
        out = out.strip()
        assert out.startswith("```bash")
        return out.split("\n")[1:-1]




# Prompt that runs the bash command.


class BashPrompt(Prompt[List[str], str]):
    def prompt(self, inp: List[str]) -> str:
        return ";".join(inp).replace("\n", "")

    def parse(self, out: str, inp: List[str]) -> str:
        return out




with start_chain("bash") as backend:
    question = '"go up one directory, and then into the minichain directory, and list the files in the directory"'
    prompt = CLIPrompt(backend.OpenAI()).chain(BashPrompt(backend.BashProcess()))
    result = prompt({"question": question})
    print(result)

# + tags=["hide_inp"]
CLIPrompt().show(
    {"question": "list the files in the directory"}, """```bash\nls\n```"""
)
# -

# + tags=["hide_inp"]
BashPrompt().show(["ls", "cat file.txt"], "hello")
# -
    
show_log("bash.log")
