# + tags=["hide_inp"]

desc = """
### Agent

Chain that executes different tools based on model decisions. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/bash.ipynb)

(Adapted from LangChain )
"""
# -

# $

from minichain import Id, prompt, OpenAI, show, transform, Mock, Break
from gradio_tools.tools import StableDiffusionTool, ImageCaptioningTool, ImageToMusicTool


# class ImageCaptioningTool:
#     def run(self, inp):
#         return "This is a picture of a smiling huggingface logo."

#     description = "Image Captioning"

tools = [StableDiffusionTool(), ImageCaptioningTool(), ImageToMusicTool()]


@prompt(OpenAI(stop=["Observation:"]),
        template_file="agent.pmpt.tpl")
def agent(model, query, history):
    return model(dict(tools=[(str(tool.__class__.__name__), tool.description)
                             for tool in tools],
                      input=query,
                      agent_scratchpad=history
                      ))
@transform()
def tool_parse(out):
    lines = out.split("\n")
    if lines[0].split("?")[-1].strip() == "Yes":
        tool = lines[1].split(":", 1)[-1].strip()
        command = lines[2].split(":", 1)[-1].strip()
        return tool, command
    else:
        return Break()

@prompt(tools)
def tool_use(model, usage):
    selector, command = usage
    for i, tool in enumerate(tools):
        if selector == tool.__class__.__name__:
            return model(command, tool_num=i)
    return ("",)

@transform()
def append(history, new, observation):
    return history + "\n" + new + "Observation: " + observation

def run(query):
    history = ""
    observations = []
    for i in range(3):
        select_input = agent(query, history)
        observations.append(tool_use(tool_parse(select_input)))
        history = append(history, select_input, observations[i])

    return observations[-1]

# $

gradio = show(run,
              subprompts=[agent, tool_use] * 3,
              examples=[
                  "I would please like a photo of a dog riding a skateboard. "
                  "Please caption this image and create a song for it.",
                  'Use an image generator tool to draw a cat.',
                  'Caption the image https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png from the internet'],
              out_type="markdown",
              description=desc,
              show_advanced=False
              )
if __name__ == "__main__":
    gradio.queue().launch()

