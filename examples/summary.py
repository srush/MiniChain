# Summarize a long document by chunking and summarizing parts.  Uses
# aynchronous calls to the API.  Adapted from LangChain [Map-Reduce
# summary](https://langchain.readthedocs.io/en/stable/_modules/langchain/chains/mapreduce.html).

import trio

from minichain import TemplatePrompt, show_log, start_chain

# Prompt that asks LLM to produce a bash command.


class SummaryPrompt(TemplatePrompt):
    template_file = "summary.pmpt.tpl"


def chunk(f, width=4000, overlap=800):
    "Split a documents into 4800 character overlapping chunks"
    text = open(f).read().replace("\n\n", "\n")
    chunks = []
    for i in range(4):
        if i * width > len(text):
            break
        chunks.append({"text": text[i * width : (i + 1) * width + overlap]})
    return chunks


with start_chain("summary") as backend:
    prompt = SummaryPrompt(backend.OpenAI())
    list_prompt = prompt.map()

    # Map - Summarize each chunk in parallel
    out = trio.run(list_prompt.arun, chunk("../state_of_the_union.txt"))

    # Reduce - Summarize the summarized chunks
    print(prompt({"text": "\n".join(out)}))

# + tags=["hide_inp"]
SummaryPrompt().show(
    {"text": "One way to fight is to drive down wages and make Americans poorer."},
    "Make Americans poorer",
)
# -

show_log("summary.log")
