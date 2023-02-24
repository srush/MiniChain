from typing import List

from minichain import JinjaPrompt, Prompt, start_chain

# Prompt that asks LLM to produce a bash command.


class CLIPrompt(JinjaPrompt[List[str]]):
    template_file = "bash.pmpt.tpl"

    def parse(self, out: str, inp: JinjaPrompt.IN) -> List[str]:
        assert result.startswith("```bash")
        return result.split("\n")[1:-1]


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
    mock = backend.Mock(
        [
            """```bash
ls
ls
ls
```"""
        ]
    )
    # openai = backend.OpenAI()
    question = '"go up one directory and list the files in the directory"'
    prompt = CLIPrompt(mock).chain(BashPrompt(backend.BashProcess()))
    result = prompt({"question": question})
    print(result)
