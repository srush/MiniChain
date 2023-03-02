# Mini-Chain

A tiny library for **large** language models.

[[Documentation and Examples](https://srush.github.io/MiniChain)]

<center><img width="200px" src="https://user-images.githubusercontent.com/35882/218286642-67985b6f-d483-49be-825b-f62b72c469cd.png"/></center>

Write apps that can easily and efficiently call multiple language models.

* Code ([math.py](https://github.com/srush/MiniChain/blob/main/examples/math.py)):

```python
# A prompt from the Jinja template below.
class MathPrompt(TemplatePrompt[str]):
    template_file = "math.pmpt.tpl"

with start_chain("math") as backend:
    # MathPrompt with OpenAI backend
    p1 = MathPrompt(backend.OpenAI())
    # A prompt that simply runs Python
    p2 = SimplePrompt(backend.Python())
    # Chain them together
    prompt = p1.chain(p2)
    # Call chain with a question.
    question ="'What is the sum of the powers of 3 (3^i) that are smaller than 100?"
    print(prompt({"question": question}))
```

* Template ([math.pmpt.tpl](https://github.com/srush/MiniChain/blob/main/examples/math.pmpt.tpl)):

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

* Install and Execute:

```bash
> pip install git+https://github.com/srush/MiniChain/
> export OPENAI_KEY="sk-***"
> python math.py
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
        memory = self.memory if len(self.memory) < MEMORY else self.memory[1:]
        return State(memory + [(self.human_input, response)])
```

See the full [Chat](https://srush.github.io/MiniChain/examples/chatgpt/) example.
It keeps track of the last two responses that it has seen.


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

class KNNPrompt(EmbeddingPrompt):
    def find(self, out, inp):
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

### Typed Prompt

MiniChain can automatically generate a prompt header for you that aims to ensure the
output follows a given typed specification. For example, if you run the following code
MiniChain will produce prompt that returns a list of `Color` objects.

```python
class ColorType(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

@dataclass
class Color:
    color: ColorType
    object: str
    explanation: str


class ColorPrompt(minichain.TypedTemplatePrompt):
    template_file = "color.pmpt.tpl"
    Out = Color
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

