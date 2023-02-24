# Mini-Chain

A tiny library for **large** language models.

[[Documentation and Examples](https://srush.github.io/minichain)]

<center><img width="200px" src="https://user-images.githubusercontent.com/35882/218286642-67985b6f-d483-49be-825b-f62b72c469cd.png"/></center>

* Code:

```python
class MathPrompt(JinjaPrompt[str]):
    template_file = "math.pmpt.tpl"

with start_chain("math") as backend:
    prompt = MathPrompt(backend.OpenAI()).chain(SimplePrompt(backend.Python()))
    question ="'What is the sum of the powers of 3 (3^i) that are smaller than 100?"
    result = prompt({"question": question})
    print(result)
```

* Template:

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

# Tutorial

Mini-Chain defines one class called `Prompt[Input, Output]`. 
It wraps a call to a language model in a type-safe way. `Input` is the input 
to the function and `Output` is what it returns. 

You can write your own prompts by overriding the `prompt` and `parse`
function to return a typed result. 

```python
class MyPrompt(Prompt[str, bool]):
    def parse(inp: str) -> str:
        return f"Answer 'yes' if this is a color, {inp}. Answer:"
    
    def parse(out: str, inp) -> bool:
        # Encode the parsing logic
        return out == "Yes" 
```

`Prompt` is instantiated by passing a `Backend`. The library is mainly
designed to use OpenAI but there are also backends for Google Search,
and random tools like bash and python.


```python
with start_chain("mychain") as backend:
    prompt1 = MyPrompt(backend.OpenAI())
    if prompt1("blue"):
        ...
```

You can write a type-safe Python program using these prompts in the
standard manner, or you can chain together prompts using function
composition. Specifically, turns `Prompt[Input, Mid]` and `Prompt[Mid,
Output]` to `Prompt[Input, Output]`.

```python
with start_chain("mychain") as backend:
    prompt0 = SimplePrompt(backend.OpenAI())
    chained_prompt = prompt0.chain(prompt1)
    if chained_prompt("..."):
        ...
```

Prompt `SimplePrompt` simply passes its input string to the
language-model and returns its output string.  Often the simplest way
to use prompts is through the predefined `JinjaPrompt[Output]`.  You
provide it with a
[Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) which
it uses to prompt the language model. You can also customize it's
output type through a `parse` method.

```python
class MathPrompt(JinjaPrompt[str]):
    template_file = "math.pmpt.tpl"
```

To view the output of your program we use the
[eliot](https://eliot.readthedocs.io/en/stable/) logging framework.
You can run the following command to get the full output of your
system.

```bash
eliot-tree -l 0 mychain.log
```

## Advanced: Asynchronous Calls

One benefit of wrapping prompts in this way is that it allows for
asynchronous execution. We use the
[trio](https://trio.readthedocs.io/en/stable/) library for
asynchronous running of prompts. Prompt has a method `arun` which will
make the language model call asynchronous.

```python
async def fn1(prompt1):
        if await prompt1.arun("blue"):
        ...

trio.run(prompt1)
```

A convenient construct is the `map` function which
converts a `Prompt[Input, Output]` to `Prompt[Sequence[Input],
Sequence[Output]]`. This method is particularly useful in conjunction
with asynchronous execution. For example this code runs a
summarization prompt with asynchonous calls to the API.


```python
with start_chain("summary") as backend:
    list_prompt = SummaryPrompt(backend.OpenAI()).map()
    out = trio.run(list_prompt.arun, documents)
```



## Advanced: Parsing

Another benefit of prompting is that it isolates text parsing from execution.
For example, it allows you to integrate declarative parsers of the output of the 
language model. 

In the SelfAsk example, there are two possible outputs from the
language model.  To make this convenient to parse, we use a parser
combinator [parsita](https://parsita.drhagen.com/) to check for both
possible outputs in a readable way with error checking.


```python
class SelfAsk(JinjaPrompt[IntermediateState | FinalState]):
    template_file = "selfask.pmpt.tpl"

    class Parser(TextParsers):
        follow = (lit("Follow up:") >> reg(r".*")) > IntermediateState
        finish = (lit("So the final answer is: ") >> reg(r".*")) > FinalState
        response = follow | finish

    def parse(self, response: str, inp):
        return self.Parser.response.parse(response).or_die()
```
