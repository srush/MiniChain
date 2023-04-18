# + tags=["hide_inp"]
desc = """
### Question Answering with Retrieval

Chain that answers questions with embeedding based retrieval. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/qa.ipynb)

(Adapted from [OpenAI Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).)
"""
# -

# $

import datasets
import numpy as np
from minichain import prompt, transform, show, OpenAIEmbed, OpenAI
from manifest import Manifest

# We use Hugging Face Datasets as the database by assigning
# a FAISS index.

olympics = datasets.load_from_disk("olympics.data")
olympics.add_faiss_index("embeddings")


# Fast KNN retieval prompt

@prompt(OpenAIEmbed())
def embed(model, inp):
    return model(inp)

@transform()
def get_neighbors(inp, k):
    res = olympics.get_nearest_examples("embeddings", np.array(inp), k)
    return res.examples["content"]

@prompt(OpenAI(), template_file="qa.pmpt.tpl")
def get_result(model, query, neighbors):
    return model(dict(question=query, docs=neighbors))

def qa(query):
    n = get_neighbors(embed(query), 3)
    return get_result(query, n)

# $


questions = ["Who won the 2020 Summer Olympics men's high jump?",
             "Why was the 2020 Summer Olympics originally postponed?",
             "In the 2020 Summer Olympics, how many gold medals did the country which won the most medals win?",
             "What is the total number of medals won by France?",
             "What is the tallest mountain in the world?"]

gradio = show(qa,
              examples=questions,
              subprompts=[embed, get_result],
              description=desc,
              code=open("qa.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )
if __name__ == "__main__":
    gradio.queue().launch()

