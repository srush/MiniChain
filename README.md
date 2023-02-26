# Mini-Chain

A tiny library for **large** language models.

[[Documentation and Examples](https://srush.github.io/MiniChain)]

<center><img width="200px" src="https://user-images.githubusercontent.com/35882/218286642-67985b6f-d483-49be-825b-f62b72c469cd.png"/></center>

Write apps that can easily and efficiently call multiple language models.

* Code (`math.py`):

```python
class MathPrompt(TemplatePrompt[str]):
    template_file = "math.pmpt.tpl"

with start_chain("math") as backend:
    prompt = MathPrompt(backend.OpenAI()).chain(SimplePrompt(backend.Python()))
    question ="'What is the sum of the powers of 3 (3^i) that are smaller than 100?"
    result = prompt({"question": question})
    print(result)
```

* Template (`math.pmpt.tpl`):

```
...
Question:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Code:
2 + 2/2

Question:
{{question}}
Code:
```

* Execution:

```bash
> pip install git+https://github.com/srush/MiniChain/
> export OPENAI_KEY="sk-***"
> python math.py
```

## Examples

This library allows us to implement several popular approaches in a few lines of code.

* [Retrieval-Augemented QA](https://srush.github.io/MiniChain/examples/qa/)
* [PAL](https://srush.github.io/MiniChain/examples/pal/) - [(Gao et al 2023)](https://arxiv.org/pdf/2211.10435.pdf)
* [Self-Ask](https://srush.github.io/MiniChain/examples/selfask/) - [(Press et al 2022)](https://ofir.io/self-ask.pdf)
* [Chain-of-Thought](https://srush.github.io/MiniChain/examples/bash/) - [(Wei et al 2022)](https://arxiv.org/abs/2201.11903)

It supports the current backends.

* OpenAI (Completions / Embeddings)
* Hugging Face 🤗
* Google Search
* Python
* Manifest-ML (AI21, Cohere, Together)
* Bash

## Tutorial

Mini-chain is based on Prompts.

![image](https://user-images.githubusercontent.com/35882/221280012-d58c186d-4da2-4cb6-96af-4c4d9069943f.png)

You can write your own prompts by overriding the `prompt` and `parse`
function on the `Prompt[Input, Output]` class.

```python
class ColorPrompt(Prompt[str, bool]):
    def prompt(inp: str) -> str:
        "Encode prompting logic"
        return f"Answer 'Yes' if this is a color, {inp}. Answer:"

    def parse(out: str, inp) -> bool:
        # Encode the parsing logic
        return out == "Yes"
```

The LLM for the Prompt is specified by the backend. To run a prompt, we give a backend and then call it like a function. To access backends, you need to call `start_chain`
which also manages logging.

```python
with start_chain("color") as backend:
    prompt1 = ColorPrompt(backend.OpenAI())
    if prompt1("blue"):
        print("It's a color!")
```

You can write a standard Python program just by calling these prompts. Alternatively you can chain prompts together.

![image](https://user-images.githubusercontent.com/35882/221281771-3770be96-02ce-4866-a6f8-c458c9a11c6f.png)


```python
with start_chain("mychain") as backend:
    prompt0 = SimplePrompt(backend.OpenAI())
    chained_prompt = prompt0.chain(prompt1)
    if chained_prompt("..."):
        ...
```

Prompt `SimplePrompt` simply passes its input string to the
language-model and returns its output string.

We also include `TemplatePrompt[Output]` which assumes `parse` uses template from the
[Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) language.

```python
class MathPrompt(TemplatePrompt[str]):
    template_file = "math.pmpt.tpl"
```

Logging is done automatically based on the name of your chain using the [eliot](https://eliot.readthedocs.io/en/stable/) logging framework.
You can run the following command to get the full output of your
system.

```python
show_log("mychain.log")
```


### Documents and Embeddings

MiniChain is agnostic to how you manage documents and embeddings. We recommend using
the [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) library with
built in FAISS indexing.

![image](https://user-images.githubusercontent.com/35882/221387303-e3dd8456-a0f0-4a70-a1bb-657fe2240862.png)


Here is the implementation.

```python
# Load and index a dataset
olympics = datasets.load_from_disk("olympics.data")
olympics.add_faiss_index("embeddings")

class KNNPrompt(Prompt):
    def parse(self, out, inp):
        return olympics.get_nearest_examples("embeddings", np.array(out), 3)
```



This creates a K-nearest neighbors (KNN) `Prompt` that looks up the
3 closest documents based on embeddings of the question asked.
See the full [Retrieval-Augemented QA](https://srush.github.io/MiniChain/examples/qa/)
example.


We recommend creating these embeddings offline using the batch map functionality of the
datasets library.

```python
def embed(x):
    emb = openai.Embedding.create(input=x["content"], engine=EMBEDDING_MODEL)
    return {"embeddings": [np.array(emb['data'][i]['embedding'])
                           for i in range(len(emb["data"]))]}
x = dataset.map(embed, batch_size=BATCH_SIZE, batched=True)
x.save_to_disk("olympics.data")
```

There are other ways to do this such as [sqllite](https://github.com/asg017/sqlite-vss)
or [Weaviate](https://weaviate.io/).

## Advanced

### Asynchronous Calls

Prompt chains make it easier to manage asynchronous execution. Prompt has a method `arun` which will
make the language model call asynchronous.
Async calls need the [trio](https://trio.readthedocs.io/en/stable/) library.

```python
import trio
async def fn1(prompt1):
        if await prompt1.arun("blue"):
        ...

trio.run(prompt1)
```

A convenient construct is the `map` function which runs a prompt on a list of inputs.

![image](https://user-images.githubusercontent.com/35882/221283494-6f76ee85-3652-4bb3-bc42-4e961acd1477.png)


This code runs a summarization prompt with asynchonous calls to the API.


```python
with start_chain("summary") as backend:
    list_prompt = SummaryPrompt(backend.OpenAI()).map()
    out = trio.run(list_prompt.arun, documents)
```



### Parsing

Minichain lets you use whatever parser you would like.
One example is [parsita](https://parsita.drhagen.com/) a
cool parser combinator library. This example builds a little
state machine based on the LLM response with error handling.

```python
class SelfAsk(TemplatePrompt[IntermediateState | FinalState]):
    template_file = "selfask.pmpt.tpl"

    class Parser(TextParsers):
        follow = (lit("Follow up:") >> reg(r".*")) > IntermediateState
        finish = (lit("So the final answer is: ") >> reg(r".*")) > FinalState
        response = follow | finish

    def parse(self, response: str, inp):
        return self.Parser.response.parse(response).or_die()
```

[[Full Examples](https://srush.github.io/MiniChain)]

