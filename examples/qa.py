# + tags=["hide_inp"]
desc = """
# QA

Questions answering with embeddings.  Adapted from [OpenAI Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).
"""
# -

import datasets
import numpy as np

from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain

# We use Hugging Face Datasets as the database by assigning
# a FAISS index.

olympics = datasets.load_from_disk("olympics.data")
olympics.add_faiss_index("embeddings")


# Fast KNN retieval prompt


class KNNPrompt(EmbeddingPrompt):
    def find(self, out, inp):
        res = olympics.get_nearest_examples("embeddings", np.array(out), 3)
        return {"question": inp, "docs": res.examples["content"]}


# QA prompt to ask question with examples


class QAPrompt(TemplatePrompt):
    template_file = "qa.pmpt.tpl"


with start_chain("qa") as backend:
    prompt = KNNPrompt(backend.OpenAIEmbed()).chain(QAPrompt(backend.OpenAI()))
    
question = "Who won the 2020 Summer Olympics men's high jump?"

gradio = prompt.to_gradio(fields=["query"],
                          examples=[question],
                          description=desc)
if __name__ == "__main__":
    gradio.launch()



# # + tags=["hide_inp"]
# QAPrompt().show(
#     {"question": "Who won the race?", "docs": ["doc1", "doc2", "doc3"]}, "Joe Bob"
# )
# # -

# show_log("qa.log")
