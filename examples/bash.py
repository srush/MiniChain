# Notebook to generate and run a bash command.
# Adapted from LangChain
# [BashChain](https://langchain.readthedocs.io/en/latest/modules/chains/examples/llm_bash.html)

import minichain

# Prompt that asks LLM to produce a bash command.


class CLIPrompt(minichain.TemplatePrompt):
    template_file = "bash.pmpt.tpl"

    def parse(self, out: str, inp):
        out = out.strip()
        assert out.startswith("```bash")
        return out.split("\n")[1:-1]


# Prompt that runs the bash command.


class BashPrompt(minichain.Prompt):
    def prompt(self, inp) -> str:
        return ";".join(inp).replace("\n", "")

    def parse(self, out: str, inp) -> str:
        return out


# Generate and run bash command.

with minichain.start_chain("bash") as backend:
    question = (
        '"go up one directory, and then into the minichain directory,'
        'and list the files in the directory"'
    )
    prompt = CLIPrompt(backend.OpenAI()).chain(BashPrompt(backend.BashProcess()))
    result = prompt({"question": question})
    print(result)

# View the prompts.

# + tags=["hide_inp"]
CLIPrompt().show(
    {"question": "list the files in the directory"}, """```bash\nls\n```"""
)
# -


# + tags=["hide_inp"]
BashPrompt().show(["ls", "cat file.txt"], "hello")
# -

# View the run log.

minichain.show_log("bash.log")
