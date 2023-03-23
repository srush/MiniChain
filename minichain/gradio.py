import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import gradio as gr
from gradio.blocks import Block
from jinja2 import Environment, PackageLoader, select_autoescape

from minichain import start_chain

from .base import Prompt

CSS = """
#clean div.form {border: 0px}
#response {border: 0px; background: #ffeec6}
#prompt {border: 0px;background: aliceblue}
#json {border: 0px}
#result {border: 0px; background: #c5e0e5}
#inner {margin: 10px; padding: 10px; font-size: 20px; }
#inner textarea {border: 0px}
div.gradio-container {color: black}
span.head {font-size: 60pt; font-family: cursive;}
body {
  --text-sm: 15px;
  --text-md: 20px;
  --text-lg: 22px;
  --input-text-size: 20px;
  --section-text-size: 20px;
}
"""


@dataclass
class HTML:
    html: str

    def _repr_html_(self) -> str:
        return self.html


@dataclass
class DisplayOptions:
    markdown: bool = True


all_data = gr.State({})
final_output = gr.State({})


@dataclass
class Constructor:
    fns: List[Callable[[Dict[Block, Any]], Dict[Block, Any]]] = field(
        default_factory=list
    )
    inputs: Set[Block] = field(default_factory=set)
    outputs: Set[Block] = field(default_factory=set)

    def merge(self, other: "Constructor") -> "Constructor":
        return Constructor(
            self.fns + other.fns,
            self.inputs | other.inputs,
            self.outputs | other.outputs,
        )

    def add_inputs(self, inputs: List[Block]) -> "Constructor":
        return Constructor(self.fns, self.inputs | set(inputs), self.outputs)

    def fn(self, data: Dict[Block, Any]) -> Dict[Block, Any]:
        out: Dict[Block, Any] = {}
        for fn in self.fns:
            out = {**out, **fn(data)}
        return out


def to_gradio_block(
    base_prompt: Prompt[Any, Any, Any],
    i: int,
    display_options: DisplayOptions = DisplayOptions(),
) -> Constructor:

    with gr.Accordion(label=f"👩  Prompt: {str(base_prompt._fn)}", elem_id="prompt"):
        if display_options.markdown:
            prompt = gr.Markdown(label="", elem_id="inner")
        else:
            prompt = gr.Textbox(label="", elem_id="inner")
    with gr.Accordion(label="💻", elem_id="response"):
        if display_options.markdown:
            result = gr.Markdown(label="", elem_id="inner")
        else:
            result = gr.Textbox(label="", elem_id="inner")

    with gr.Accordion(label="...", elem_id="json", open=False):
        backend = gr.Markdown(f"Backend: {base_prompt.backend}", elem_id="json")
        input = gr.JSON(elem_id="json", label="Input")
        json = gr.JSON(elem_id="json", label="Output")
        trial = gr.JSON(elem_id="json", label="Previous Trial")

        if base_prompt.template_file:
            # gr.Markdown(f"<center>{base_prompt.template_file}</center>")
            c = gr.Code(
                label=f"Template: {base_prompt.template_file}",
                value=open(base_prompt.template_file).read(),
                elem_id="inner",
            )
            # btn = gr.Button("Modify Template")
            # if base_prompt.template_file is not None:

            #     def update_template(template: str) -> None:
            #         if base_prompt.template_file is not None:
            #             with open(base_prompt.template_file, "w") as doc:
            #                 doc.write(template)

            #     btn.click(update_template, inputs=c)

    def update(data: Dict[Block, Any]) -> Dict[Block, Any]:
        prev_request_ = ""
        if (base_prompt._id, i - 1) in data[all_data]:
            prev_request_ = data[all_data][base_prompt._id, i - 1][-1][1].prompt
        trials = len(data[all_data][base_prompt._id, i])
        input_, request_, response_, output_ = data[all_data][base_prompt._id, i][-1]

        def format(s: Any) -> Any:
            if isinstance(s, str):
                return {"string": s}
            return s

        def mark(s: Any) -> Any:
            return s  # f"```text\n{s}\n```"

        j = 0
        for (a, b) in zip(request_.prompt, prev_request_):
            if a != b:
                break
            j += 1

        if j > 30:
            new_prompt = "...\n" + request_.prompt[j:]
        else:
            new_prompt = request_.prompt

        if trials == 1:
            previous_trials = []
        else:
            trial_input_, trial_request_, trial_response_, trial_output_ = data[
                all_data
            ][base_prompt._id, i][trials - 2]
            previous_trials = {
                "input": trial_input_,
                "prompt": trial_request_.prompt,
                "response": trial_response_,
                "output": trial_output_,
            }
        ret = {
            input: format(input_),
            prompt: mark(new_prompt),
            result: mark(response_),
            json: format(output_),
            trial: previous_trials,
        }
        return ret

    return Constructor([update], set(), {input, prompt, result, json, trial})


def chain_blocks(prompts: List[Prompt[Any, Any, Any]]) -> Constructor:
    cons = Constructor()
    count = {}
    for p in prompts:
        count.setdefault(p._id, 0)
        i = count[p._id]
        cons = cons.merge(to_gradio_block(p, i))
        count[p._id] += 1
    return cons


def api_keys(keys: Set[str] = {"OPENAI_API_KEY"}) -> None:
    if all([k in os.environ for k in keys]):
        return
    key_names = {}

    with gr.Accordion(label="API Keys", elem_id="json", open=False):
        if "OPENAI_API_KEY" in keys and "OPENAI_API_KEY" not in os.environ:
            key_names["OPENAI_API_KEY"] = gr.Textbox(
                os.environ.get("OPENAI_API_KEY"),
                label="OpenAI Key",
                elem_id="json",
                type="password",
            )
            gr.Markdown(
                """
            * [OpenAI Key](https://platform.openai.com/account/api-keys)
            """
            )

        if "HF_KEY" in keys:
            gr.Markdown(
                """
            * [Hugging Face Key](https://huggingface.co/settings/tokens)
            """
            )

            key_names["HF_KEY"] = gr.Textbox(
                os.environ.get("HF_KEY"),
                label="Hugging Face Key",
                elem_id="inner",
                type="password",
            )

        if "SERP_KEY" in keys:
            gr.Markdown(
                """
            * [Search Key](https://serpapi.com/users/sign_in)
            """
            )
            key_names["SERP_KEY"] = gr.Textbox(
                os.environ.get("SERP_KEY"),
                label="Search Key",
                elem_id="inner",
                type="password",
            )

        api_btn = gr.Button("Save")

    def api_run(data):  # type: ignore
        for k, v in key_names.items():
            if data[v] is not None and data[v] != "":
                os.environ[k] = data[v]
        return {}

    api_btn.click(api_run, inputs=set(key_names.values()))


def show(
    prompt: Prompt[Any, Any, Any],
    examples: Union[List[str], List[Tuple[str]]] = [""],
    subprompts: List[Prompt[Any, Any, Any]] = [],
    fields: List[str] = [],
    initial_state: Any = None,
    out_type: str = "markdown",
    keys: Set[str] = {"OPENAI_API_KEY"},
    description: str = "",
    code: str = "",
    templates: List[str] = [],
) -> gr.Blocks:
    """
    Constructs a gradio component to show a prompt chain.

    Args:
        prompt: A prompt or prompt chain to display.
        examples: A list of example inputs, either string or tuples of fields
        fields: The names of the field input to the prompt.
        initial_state: For stateful prompts, the initial value.
    """
    fields = [arg for arg in inspect.getargspec(prompt).args if arg != "state"]
    with gr.Blocks(css=CSS, theme=gr.themes.Monochrome()) as demo:
        # API Keys
        api_keys()

        constructor = Constructor()

        # Collect all the inputs
        state = gr.State(initial_state)
        constructor = constructor.merge(Constructor(inputs={state}, outputs={state}))

        # Show the description
        gr.Markdown(description)

        # Build the top query box with one input for each field.
        inputs = list([gr.Textbox(label=f) for f in fields])
        examples = gr.Examples(examples=examples, inputs=inputs)
        query_btn = gr.Button(label="Run")
        constructor = constructor.add_inputs(inputs)

        # Intermediate prompt displays
        constructor = constructor.merge(chain_blocks(subprompts))

        # Final Output result
        with gr.Accordion(label="✔️", elem_id="result"):
            typ = gr.JSON if out_type == "json" else gr.Markdown
            output = typ(elem_id="inner")

        def output_fn(data: Dict[Block, Any]) -> Dict[Block, Any]:
            final = data[final_output]
            return {state: final, output: final}

        constructor = constructor.merge(Constructor([output_fn], set(), {output}))

        def run(data):  # type: ignore
            prompt_inputs = {k: data[v] for k, v in zip(fields, inputs)}
            if initial_state is not None:
                prompt_inputs["state"] = data[state]

            with start_chain("temp") as r:
                output = prompt(**prompt_inputs).run()
            # except Exception as inst:
            #     print(inst)
            #     gr.Error(str(inst))
            #     return {}
            data[all_data] = dict(r.prompt_store)
            data[final_output] = output
            return constructor.fn(data)

        query_btn.click(run, inputs=constructor.inputs, outputs=constructor.outputs)

        if code:
            gr.Code(code, language="python", elem_id="inner")

    return demo
