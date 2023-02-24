# Generate and run a bash command.
# Adapted from LangChain
# [BashChain](https://langchain.readthedocs.io/en/latest/modules/chains/examples/llm_bash.html)

from typing import List

from minichain import JinjaPrompt, Prompt, show_log, start_chain

# Prompt that asks LLM to produce a bash command.


class CLIPrompt(JinjaPrompt[List[str]]):
    template_file = "bash.pmpt.tpl"

    def parse(self, out: str, inp: JinjaPrompt.IN) -> List[str]:
        out = out.strip()
        assert out.startswith("```bash")
        return out.split("\n")[1:-1]


CLIPrompt().show(
    {"question": "list the files in the directory"}, """```bash\nls\n```"""
)


# Prompt that runs the bash command.


class BashPrompt(Prompt[List[str], str]):
    def prompt(self, inp: List[str]) -> str:
        return ";".join(inp).replace("\n", "")

    def parse(self, out: str, inp: List[str]) -> str:
        return out


BashPrompt().show(["ls", "cat file.txt"], "hello")


with start_chain("bash") as backend:
    question = '"go up one directory, and then into the minichain directory, and list the files in the directory"'
    prompt = CLIPrompt(backend.OpenAI()).chain(BashPrompt(backend.BashProcess()))
    result = prompt({"question": question})
    print(result)


show_log("bash.log")
