# Summarize a long document by chunking and summarizing parts.
# Uses aynchronous calls to the API.
# Adapted from LangChain [Map-Reduce summary](https://langchain.readthedocs.io/en/stable/_modules/langchain/chains/mapreduce.html).

import trio

from minichain import JinjaPrompt, show_log, start_chain

# Prompt that asks LLM to produce a bash command.


class SummaryPrompt(JinjaPrompt[str]):
    template_file = "summary.pmpt.tpl"


SummaryPrompt().show(
    {
        "text": "One way to fight inflation is to drive down wages and make Americans poorer."
    },
    "Make Americans poorer",
)


def chunk(f):
    "Split a documents into 4800 character overlapping chunks"
    text = open(f).read().replace("\n\n", "\n")
    chunks = []
    W = 4000
    O = 800
    for i in range(4):
        if i * W > len(text):
            break
        chunks.append({"text": text[i * W : (i + 1) * W + O]})
    return chunks


with start_chain("summary") as backend:
    prompt = SummaryPrompt(backend.OpenAI())
    list_prompt = prompt.map()

    # Map - Summarize each chunk in parallel
    out = trio.run(list_prompt.arun, chunk("../state_of_the_union.txt"))

    # Reduce - Summarize the summarized chunks
    print(prompt({"text": "\n".join(out)}))


show_log("summary.log")
