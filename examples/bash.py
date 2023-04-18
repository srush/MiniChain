# + tags=["hide_inp"]

desc = """
### Bash Command Suggestion

Chain that ask for a command-line question and then runs the bash command. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/bash.ipynb)

(Adapted from LangChain [BashChain](https://langchain.readthedocs.io/en/latest/modules/chains/examples/llm_bash.html))
"""
# -

# $

from minichain import show, prompt, OpenAI, Bash


@prompt(OpenAI(), template_file = "bash.pmpt.tpl")
def cli_prompt(model, query):
    return model(dict(question=query))

@prompt(Bash())
def bash_run(model, x):
    x = "\n".join(x.strip().split("\n")[1:-1])
    return model(x)

def bash(query):
    return bash_run(cli_prompt(query))


# $

gradio = show(bash,
              subprompts=[cli_prompt, bash_run],
              examples=['Go up one directory, and then into the minichain directory,'
                        'and list the files in the directory',
                        "Please write a bash script that prints 'Hello World' to the console."],
              out_type="markdown",
              description=desc,
              code=open("bash.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )
if __name__ == "__main__":
    gradio.queue().launch()

