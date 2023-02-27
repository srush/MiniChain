# Questions answering with embeddings.  Adapted from [OpenAI
# Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

import datasets
import numpy as np

from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain

# Load data with embeddings (computed beforehand)

gatsby = datasets.load_from_disk("gatsby")
gatsby.add_faiss_index("embeddings")

# Fast KNN retieval prompt


class KNNPrompt(EmbeddingPrompt):
    def find(self, out, inp):
        res = gatsby.get_nearest_examples("embeddings", np.array(out), 1)
        return {"question": inp, "docs": res.examples["passages"]}


# QA prompt to ask question with examples


class QAPrompt(TemplatePrompt):
    template_file = "gatsby.pmpt.tpl"


with start_chain("gatsby") as backend:
    question = "What did Gatsby do before he met Daisy?"
    prompt = KNNPrompt(
        backend.HuggingFaceEmbed("sentence-transformers/all-mpnet-base-v2")
    ).chain(QAPrompt(backend.OpenAI()))
    result = prompt(question)
    print(result)

# + tags=["hide_inp"]
QAPrompt().show({"question": "Who was Gatsby?", "docs": ["doc1", "doc2", "doc3"]}, "")
# -

show_log("gatsby.log")
