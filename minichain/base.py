from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, Tuple

import os
import trio
from eliot import start_action
from jinja2 import Environment, PackageLoader, select_autoescape

from .backend import Backend, Request

# Type Variables.

Input = TypeVar("Input")
Input2 = TypeVar("Input2")
Output = TypeVar("Output")
Output2 = TypeVar("Output2")


# Code for visualization.
env = Environment(
    loader=PackageLoader("minichain"),
    autoescape=select_autoescape(),
    extensions=["jinja2_highlight.HighlightExtension"],
)


@dataclass
class HTML:
    html: str

    def _repr_html_(self) -> str:
        return self.html

@dataclass
class DisplayOptions:
    markdown: bool = True
    

# Main Class


class Prompt(Generic[Input, Output]):
    """`Prompt` represents a typed function from Input to Output.
    It computes its output value by calling an external services, generally
    a large language model (LLM).

    To define a new prompt, you need to inherit from this class and
    define the function `prompt` which takes a argument of type
    `Input` and produces a string to be sent to the LLM, and the
    function `parse` which takes a string and produces a value of type
    `Output`.


    To use the `Prompt` you call `__call__` or `arun` with an LLM
    backend and the arguments of the form `Input`.

    """

    def __init__(self, backend: Optional[Backend] = None,
                 data: Any = None,
                 display_options: DisplayOptions = DisplayOptions()
    ):
        self.backend = backend
        self.data = data

        self.display_options = display_options

    # START: Overloaded by the user prompts
    def prompt(self, inp: Input) -> Union[Request, str]:
        """
        Convert from the `Input` type of the function
        to a request that is sent to the backend.
        """
        return str(inp)

    def parse(self, response: str, inp: Input) -> Output:
        """
        Convert from the string response of the function
        to the output type.
        """
        return response
        # raise NotImplementedError

    def set_display_options(self, **kwargs):
        self.display_options = DisplayOptions(**kwargs)
    
    # END: Overloaded by the user prompts

    def _prompt(self, inp: Input) -> Request:
        with start_action(action_type="Input Function", input=inp):
            r = self.prompt(inp)
            if isinstance(r, str):
                return Request(r)
            else:
                return r

    def __call__(self, inp: Input) -> Output:
        return self.run_verbose(inp)[0][-1]

    def run_verbose(self, inp: Input) -> List[Tuple[Request, str, Output]]:
        assert self.backend is not None
        with start_action(action_type=str(type(self))):
            request = self._prompt(inp)
            with start_action(action_type="Prompted", prompt=request.prompt):
                result: str = self.backend.run(request)
            with start_action(action_type="Result", result=result):
                output = self.parse(result, inp)
        if not isinstance(result, str):
            result = "OBJECT"
        return [(inp, request, result, output)]
        
    
    async def arun(self, inp: Input) -> Output:
        assert self.backend is not None
        with start_action(action_type=str(type(self))):
            request = self._prompt(inp)
            with start_action(action_type="Prompted", prompt=request.prompt):
                result = await self.backend.arun(request)
            with start_action(action_type="Result", result=result):
                output = self.parse(result, inp)
        return output

    def render_prompt_html(self, inp: Input, prompt: str) -> HTML:
        return HTML(prompt.replace("\n", "<br>"))

    # Other functions.

    def show(self, inp: Input, response: str) -> HTML:
        prompt = self.prompt(inp)
        if isinstance(prompt, Request):
            prompt = prompt.prompt
        tmp = env.get_template("prompt.html.tpl")
        return HTML(
            tmp.render(
                **{
                    "name": type(self).__name__,
                    "input": str(inp),
                    "response": response,
                    "output": str(self.parse(response, inp)),
                    "prompt": self.render_prompt_html(inp, prompt).html,
                }
            )
        )

    def chain(self, other: "Prompt[Output, Output2]") -> "Prompt[Input, Output2]":
        "Chain together two prompts"
        return ChainedPrompt(self, other)

    def map(self) -> "Prompt[Sequence[Input], Sequence[Output]]":
        "Create a prompt the works on lists of inputs"
        return MapPrompt(self)


    def to_gradio_block(self):
        import gradio as gr

        # with gr.Accordion(label=type(self).__name__, open=True):
        # gr.Markdown(value=" ")
        with gr.Accordion(label=f"👩  {type(self).__name__}", elem_id="prompt"):
            if self.display_options.markdown:
                prompt = gr.Markdown(label="", elem_id="inner")
            else:
                prompt = gr.Textbox(label="", elem_id="inner")                    
        with gr.Accordion(label="💻", elem_id="response"):
            if self.display_options.markdown:
                result = gr.Markdown(label="", elem_id="inner")
            else:
                result = gr.Textbox(label="", elem_id="inner")

        with gr.Accordion(label="...", elem_id="json", open=False):
            input = gr.JSON(elem_id="json", label="Input") 
            json = gr.JSON(elem_id="json", label="Output") 

        return [input, prompt, result, json]
        
    
    def to_gradio(self, examples=[], fields=[], out_type="markdown"):
        import gradio as gr
        block = self.to_gradio_block()
        with gr.Blocks(css="#clean div.form {border: 0px} #response {border: 0px; background: #ffeec6} #prompt {border: 0px;background: aliceblue} #json {border: 0px} #result {border: 0px; background: #c5e0e5} #inner {padding: 20px} #inner textarea {border: 0px}") as demo:

            key_names = {}
            with gr.Accordion(label="API Keys", open=False): 
                key_names["OPENAI_KEY"] = gr.Textbox(os.environ.get("OPENAI_KEY"), label="OpenAI Key")
                key_names["HF_KEY"] = gr.Textbox(os.environ.get("HF_KEY"), label="Hugging Face Key")
                key_names["SERP_KEY"] = gr.Textbox(os.environ.get("SERP_KEY"), label="SERP Key")

            
            # with gr.Box(elem_id="clean"):
            if True:
                inputs = []
                
                input_names = {}
                for f in fields:
                    input_names[f] = gr.Textbox(label=f)
                inputs += input_names.values()
                examples = gr.Examples(examples=examples, inputs=list(input_names.values()))
                query_btn = gr.Button(label="Run")


            
            # query_btn = gr.Button(label="query")
            outputs = self.to_gradio_block()

            #with gr.Accordion(label="Final Result"):
            with gr.Accordion(label="✔️", elem_id="result"):
                if out_type == "json":
                    output = gr.JSON(elem_id="inner")
                else:
                    output = gr.Markdown(elem_id="inner")
            
            inputs += key_names.values()
            
            def run(data):
                for k, v in key_names.items():
                    if v != "":
                        os.environ[k] = data[v]
                    
                ls = self.run_verbose({k: data[v] for k, v in input_names.items()})
                def format(s):
                    if isinstance(s, str):
                        return {"string": s}
                    return s
                def mark(s):
                    return s# f"```text\n{s}\n```"

                return [x 
                        for (input, request, result, output) in ls
                        for x in [format(input), mark(request.prompt), mark(result), format(output)]]  + [ls[-1][-1]]
                

            outputs = outputs + [output]
            # input.submit(run, inputs=set(inputs), outputs=outputs)
            query_btn.click(run, inputs=set(inputs), outputs=outputs)            

            # with gr.Row():
            #     key = gr.Textbox(label="OpenAI Key")
           
        return demo

        
    

class ChainedPrompt(Prompt[Input, Output2]):
    def __init__(
        self, prompt1: Prompt[Input, Output], prompt2: Prompt[Output, Output2]
    ):
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def to_gradio_block(self):
        prompt1 = self.prompt1.to_gradio_block()
        prompt2 = self.prompt2.to_gradio_block()
        return prompt1 + prompt2
    
    def __call__(self, inp: Input) -> Output2:
        out = self.prompt1(inp)
        return self.prompt2(out)

    def run_verbose(self, inp: Input) -> List[Tuple[Request, str, Output]]:
        ls1 = self.prompt1.run_verbose(inp)
        ls2 = self.prompt2.run_verbose(ls1[-1][-1])
        return ls1 + ls2
    
    async def arun(self, inp: Input) -> Output2:
        out = await self.prompt1.arun(inp)
        return await self.prompt2.arun(out)


class MapPrompt(Prompt[Sequence[Input], Sequence[Output]]):
    def __init__(self, prompt: Prompt[Input, Output]):
        self.prompt1 = prompt

    def __call__(self, inp: Sequence[Input]) -> Sequence[Output]:
        return [self.prompt1(i) for i in inp]

    async def arun(self, inp: Sequence[Input]) -> Sequence[Output]:
        results: List[Optional[Output]] = [None] * len(inp)

        async def runner(i: int) -> None:
            results[i] = await self.prompt1.arun(inp[i])

        async with trio.open_nursery() as nursery:
            for i in range(len(inp)):
                nursery.start_soon(runner, i)

        ret: List[Output] = []
        for i in range(len(inp)):
            r = results[i]
            assert r is not None
            ret.append(r)
        return ret
