# Questions answering with embeddings.
# Adapted from [OpenAI Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

from minichain import JinjaPrompt, Prompt, start_chain, show_log
import datasets
import numpy as np

# Load data with embeddings (computed beforehand)

olympics = datasets.load_from_disk("olympics.data")
olympics.add_faiss_index("embeddings")

# Fast KNN retieval prompt 

class KNNPrompt(Prompt):
    def parse(self, out, inp):
        res = olympics.get_nearest_examples("embeddings",
                                            np.array(out), 3)
        return {"question": inp, "docs": res.examples["content"]}

# QA prompt to ask question with examples

class QAPrompt(JinjaPrompt):
    template_file = "qa.pmpt.tpl"
QAPrompt().show({"question": "Who won the race?", "docs": ["doc1", "doc2", "doc3"]},
                "Joe Bob")
    
with start_chain("qa") as backend:
    question = "Who won the 2020 Summer Olympics men's high jump?"
    prompt =  KNNPrompt(backend.OpenAIEmbed()).chain(QAPrompt(backend.OpenAI()))
    result = prompt(question)
    print(result)


show_log("qa.log")
