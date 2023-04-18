# + tags=["hide_inp"]
desc = """
### Book QA

Chain that does question answering with Hugging Face embeddings. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/gatsby.ipynb)

(Adapted from the [LlamaIndex example](https://github.com/jerryjliu/gpt_index/blob/main/examples/gatsby/TestGatsby.ipynb).)
"""
# -

# $

import datasets
import numpy as np
from minichain import prompt, show, HuggingFaceEmbed, OpenAI, transform

# Load data with embeddings (computed beforehand)

gatsby = datasets.load_from_disk("gatsby")
gatsby.add_faiss_index("embeddings")

# Fast KNN retrieval prompt

@prompt(HuggingFaceEmbed("sentence-transformers/all-mpnet-base-v2"))
def embed(model, inp):
    return model(inp)

@transform()
def get_neighbors(embedding, k=1):
    res = gatsby.get_nearest_examples("embeddings", np.array(embedding), k)
    return res.examples["passages"]

@prompt(OpenAI(), template_file="gatsby.pmpt.tpl")
def ask(model, query, neighbors):
    return model(dict(question=query, docs=neighbors))

def gatsby_q(query):
    n = get_neighbors(embed(query))
    return ask(query, n)


# $


gradio = show(gatsby_q,
              subprompts=[ask],
              examples=["What did Gatsby do before he met Daisy?",
                        "What did the narrator do after getting back to Chicago?"],
              keys={"HF_KEY"},
              description=desc,
              code=open("gatsby.py", "r").read().split("$")[1].strip().strip("#").strip()
              )
if __name__ == "__main__":
    gradio.queue().launch()
