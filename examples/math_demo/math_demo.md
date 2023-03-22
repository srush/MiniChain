```python tags=["hide_inp"]
desc = """
### Word Problem Solver

Chain that solves a math word problem by first generating and then running Python code. [[Code](https://github.com/srush/MiniChain/blob/main/examples/math.py)]

(Adapted from Dust [maths-generate-code](https://dust.tt/spolu/a/d12ac33169))
"""
```

```python
import minichain
```

Prompt that asks LLM for code from math.

```python
class MathPrompt(minichain.TemplatePrompt):
    template_file = "math.pmpt.tpl"
```

Ask a question and run it as python code.

```python
with minichain.start_chain("math") as backend:
    math_prompt = MathPrompt(backend.OpenAI())
    code_prompt = minichain.SimplePrompt(backend.Python())
    prompt = math_prompt.chain(code_prompt)
```


```python tags=["hide_inp"]
gradio = prompt.to_gradio(fields =["question"],
                          examples=["What is the sum of the powers of 3 (3^i) that are smaller than 100?",
                                    "What is the sum of the 10 first positive integers?",
                                    "Carla is downloading a 200 GB file. She can download 2 GB/minute, but 40% of the way through the download, the download fails. Then Carla has to restart the download from the beginning. How load did it take her to download the file in minutes?"],
                          out_type="markdown",
                          description=desc,
                          code=open("math_demo.md", "r").read()
                 
)
if __name__ == "__main__":
    gradio.launch()
#-
    
```
