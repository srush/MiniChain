# + tags=["hide_inp"]
desc = """
### Question Answering with Retrieval

Chain that answers questions with embeedding based retrieval. [[Code](https://github.com/srush/MiniChain/blob/main/examples/qa.py)]

(Adapted from [OpenAI Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).)
"""
# -

# $

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

# $

    
questions = ["Who won the 2020 Summer Olympics men's high jump?",
             "Why was the 2020 Summer Olympics originally postponed?",
             "In the 2020 Summer Olympics, how many gold medals did the country which won the most medals win?",
             "What is the total number of medals won by France?",
             "What is the tallest mountain in the world?"]

gradio = prompt.to_gradio(fields=["query"],
                          examples=questions,
                          description=desc,
                          code=open("qa.py", "r").read().split("$")[1].strip().strip("#").strip(),
                          templates=[open("qa.pmpt.tpl")]
                          )
if __name__ == "__main__":
    gradio.launch()



# # + tags=["hide_inp"]
# QAPrompt().show(
#     {"question": "Who won the race?", "docs": ["doc1", "doc2", "doc3"]}, "Joe Bob"
# )
# # -

# show_log("qa.log")
