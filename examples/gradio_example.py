# + tags=["hide_inp"]

desc = """
### Gradio Tool

Examples using the gradio tool [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/gradio_example.ipynb)

"""
# -

# $

from minichain import show, prompt, OpenAI, GradioConf
import gradio as gr
from gradio_tools.tools import StableDiffusionTool, ImageCaptioningTool

@prompt(OpenAI())
def picture(model, query):
    return model(query)

@prompt(StableDiffusionTool(),
        gradio_conf=GradioConf(
            block_output= lambda: gr.Image(),
            block_input= lambda: gr.Textbox(show_label=False)))
def gen(model, query):
    return model(query)

@prompt(ImageCaptioningTool(),
        gradio_conf=GradioConf(
            block_input= lambda: gr.Image(),
            block_output=lambda: gr.Textbox(show_label=False)))
def caption(model, img_src):
    return model(img_src)

def gradio_example(query):
    return caption(gen(picture(query)))


# $

gradio = show(gradio_example,
              subprompts=[picture, gen, caption],
              examples=['Describe a one-sentence fantasy scene.',
                        'Describe a one-sentence scene happening on the moon.'],
              out_type="markdown",
              description=desc,
              show_advanced=False
              )
if __name__ == "__main__":
    gradio.queue().launch()

