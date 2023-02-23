from typing import List

from minichain import Backend, JinjaPrompt, Prompt, start_chain


class CLIPrompt(JinjaPrompt[List[str]]):
    prompt_template = "bash.pmpt.tpl"

    def parse(self, result: str) -> List[str]:
        assert result.startswith("```bash")
        return result.split("\n")[1:-1]


class BashPrompt(Prompt[List[str], str]):
    def prompt(self, inp: List[str]) -> str:
        return ";".join(inp).replace("\n", "")

    def parse(self, out: str) -> str:
        return out


def bash(inp: str, openai: Backend, bash: Backend) -> str:
    prompt = CLIPrompt(openai).chain(BashPrompt(bash))
    return prompt({"question": inp}).val


if __name__ == "__main__":
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

        openai = backend.OpenAI("")
        result = bash(
            '"go up one directory and list the files in the directory"',
            mock,
            backend.BashProcess(),
        )
        print(result)
