<img src="https://user-images.githubusercontent.com/35882/227030644-f70e55e8-68a3-48d3-afa3-54c4de8fc210.png" width="100%">

A tiny library for coding with **large** language models. Check out the [MiniChain Zoo](https://srush-minichain.hf.space/) to get a sense of how it works.

## Coding

* Code ([math_demo.py](https://github.com/srush/MiniChain/blob/main/examples/math_demo.py)): Annotate Python functions that call language models.

```python
@prompt(OpenAI(), template_file="math.pmpt.tpl")
def math_prompt(model, question):
    "Prompt to call GPT with a Jinja template"
    return model(dict(question=question))

@prompt(Python(), template="import math\n{{code}}")
def python(model, code):
    "Prompt to call Python interpreter"
    code = "\n".join(code.strip().split("\n")[1:-1])
    return model(dict(code=code))

def math_demo(question):
    "Chain them together"
    return python(math_prompt(question))
```

* Chains ([Space](https://srush-minichain.hf.space/)): MiniChain builds a graph (think like PyTorch) of all the calls you make for debugging and error handling.
<img src="https://user-images.githubusercontent.com/35882/226965531-78df7927-988d-45a7-9faa-077359876730.png" width="50%">


```python
show(math_demo,
     examples=["What is the sum of the powers of 3 (3^i) that are smaller than 100?",
               "What is the sum of the 10 first positive integers?"],
     subprompts=[math_prompt, python],
     out_type="markdown").queue().launch()
```


* Template ([math.pmpt.tpl](https://github.com/srush/MiniChain/blob/main/examples/math.pmpt.tpl)): Prompts are separated from code.

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

* Installation

```bash
pip install minichain
export OPENAI_API_KEY="sk-***"
```

## Examples

This library allows us to implement several popular approaches in a few lines of code.

* [Retrieval-Augmented QA](https://srush.github.io/MiniChain/examples/qa/)
* [Chat with memory](https://srush.github.io/MiniChain/examples/chatgpt/)
* [Information Extraction](https://srush.github.io/MiniChain/examples/ner/)
* [Interleaved Code (PAL)](https://srush.github.io/MiniChain/examples/pal/) - [(Gao et al 2022)](https://arxiv.org/pdf/2211.10435.pdf)
* [Search Augmentation (Self-Ask)](https://srush.github.io/MiniChain/examples/selfask/) - [(Press et al 2022)](https://ofir.io/self-ask.pdf)
* [Chain-of-Thought](https://srush.github.io/MiniChain/examples/bash/) - [(Wei et al 2022)](https://arxiv.org/abs/2201.11903)

It supports the current backends.

* OpenAI (Completions / Embeddings)
* Hugging Face 🤗
* Google Search
* Python
* Manifest-ML (AI21, Cohere, Together)
* Bash

## Why Mini-Chain?

There are several very popular libraries for prompt chaining,
notably: [LangChain](https://langchain.readthedocs.io/en/latest/),
[Promptify](https://github.com/promptslab/Promptify), and
[GPTIndex](https://gpt-index.readthedocs.io/en/latest/reference/prompts.html).
These library are useful, but they are extremely large and
complex. MiniChain aims to implement the core prompt chaining
functionality in a tiny digestable library.


## Tutorial

Mini-chain is based on annotating functions as prompts.

![image](https://user-images.githubusercontent.com/35882/221280012-d58c186d-4da2-4cb6-96af-4c4d9069943f.png)


```python
@prompt(OpenAI())
def color_prompt(model, input):
    return model(f"Answer 'Yes' if this is a color, {input}. Answer:")
```

Prompt functions act like python functions, except they are lazy to access the result you need to call `run()`.

```python
if color_prompt("blue").run() == "Yes":
    print("It's a color")
```
Alternatively you can chain prompts together. Prompts are lazy, so if you want to manipulate them you need to add `@transform()` to your function. For example:

```python
@transform()
def said_yes(input):
    return input == "Yes"
```

![image](https://user-images.githubusercontent.com/35882/221281771-3770be96-02ce-4866-a6f8-c458c9a11c6f.png)

```python
@prompt(OpenAI())
def adjective_prompt(model, input):
    return model(f"Give an adjective to describe {input}. Answer:")
```


```python
adjective = adjective_prompt("rainbow")
if said_yes(color_prompt(adjective)).run():
    print("It's a color")
```


We also include an argument `template_file` which assumes model uses template from the
[Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) language.
This allows us to separate prompt text from the python code.

```python
@prompt(OpenAI(), template_file="math.pmpt.tpl")
def math_prompt(model, question):
    return model(dict(question=question))
```

### Visualization

MiniChain has a built-in prompt visualization system using `Gradio`.
If you construct a function that calls a prompt chain you can visualize it
by calling `show` and `launch`. This can be done directly in a notebook as well.

```python
show(math_demo,
     examples=["What is the sum of the powers of 3 (3^i) that are smaller than 100?",
              "What is the sum of the 10 first positive integers?"],
     subprompts=[math_prompt, python],
     out_type="markdown").queue().launch()
```


### Memory

MiniChain does not build in an explicit stateful memory class. We recommend implementing it as a queue.

![image](https://user-images.githubusercontent.com/35882/221622653-7b13783e-0439-4d59-8f57-b98b82ab83c0.png)

Here is a class you might find useful to keep track of responses.

```python
@dataclass
class State:
    memory: List[Tuple[str, str]]
    human_input: str = ""

    def push(self, response: str) -> "State":
        memory = self.memory if len(self.memory) < MEMORY_LIMIT else self.memory[1:]
        return State(memory + [(self.human_input, response)])
```

See the full Chat example.
It keeps track of the last two responses that it has seen.

### Tools and agents.

MiniChain does not provide `agents` or `tools`. If you want that functionality you can use the `tool_num` argument of model which allows you to select from multiple different possible backends. It's easy to add new backends of your own (see the GradioExample).

```python
@prompt([Python(), Bash()])
def math_prompt(model, input, lang):
    return model(input, tool_num= 0 if lang == "python" else 1)
```

### Documents and Embeddings

MiniChain does not manage documents and embeddings. We recommend using
the [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) library with
built in FAISS indexing.

![image](https://user-images.githubusercontent.com/35882/221387303-e3dd8456-a0f0-4a70-a1bb-657fe2240862.png)


Here is the implementation.

```python
# Load and index a dataset
olympics = datasets.load_from_disk("olympics.data")
olympics.add_faiss_index("embeddings")

@prompt(OpenAIEmbed())
def get_neighbors(model, inp, k):
    embedding = model(inp)
    res = olympics.get_nearest_examples("embeddings", np.array(embedding), k)
    return res.examples["content"]
```

This creates a K-nearest neighbors (KNN) prompt that looks up the
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


### Typed Prompts

MiniChain can automatically generate a prompt header for you that aims to ensure the
output follows a given typed specification. For example, if you run the following code
MiniChain will produce prompt that returns a list of `Player` objects.

```python
class StatType(Enum):
    POINTS = 1
    REBOUNDS = 2
    ASSISTS = 3

@dataclass
class Stat:
    value: int
    stat: StatType

@dataclass
class Player:
    player: str
    stats: List[Stat]


@prompt(OpenAI(), template_file="stats.pmpt.tpl", parser="json")
def stats(model, passage):
    out = model(dict(passage=passage, typ=type_to_prompt(Player)))
    return [Player(**j) for j in out]
```

Specifically it will provide your template with a string `typ` that you can use. For this example the string will be of the following form:


```
You are a highly intelligent and accurate information extraction system. You take passage as input and your task is to find parts of the passage to answer questions.

You need to output a list of JSON encoded values

You need to classify in to the following types for key: "color":

RED
GREEN
BLUE


Only select from the above list, or "Other".⏎


You need to classify in to the following types for key: "object":⏎

String



You need to classify in to the following types for key: "explanation":

String

[{ "color" : "color" ,  "object" : "object" ,  "explanation" : "explanation"}, ...]

Make sure every output is exactly seen in the document. Find as many as you can.
```

This will then be converted to an object automatically for you.


