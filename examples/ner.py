# + tags=["hide_inp"]

desc = """
### Named Entity Recognition

Chain that does named entity recognition with arbitrary labels. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/MiniChain/blob/master/examples/ner.ipynb)

(Adapted from [promptify](https://github.com/promptslab/Promptify/blob/main/promptify/prompts/nlp/templates/ner.jinja)).
"""
# -

# $

from minichain import prompt, transform, show, OpenAI
import json

@prompt(OpenAI(), template_file = "ner.pmpt.tpl")
def ner_extract(model, kwargs):
    return model(kwargs)

@transform()
def to_json(chat_output):
    return json.loads(chat_output)

@prompt(OpenAI())
def team_describe(model, inp):
    query = "Can you describe these basketball teams? " + \
        " ".join([i["E"] for i in inp if i["T"] =="Team"])
    return model(query)


def ner(text_input, labels, domain):
    extract = to_json(ner_extract(dict(text_input=text_input, labels=labels, domain=domain)))
    return team_describe(extract)


# $

gradio = show(ner,
              examples=[["An NBA playoff pairing a year ago, the 76ers (39-20) meet the Miami Heat (32-29) for the first time this season on Monday night at home.", "Team, Date", "Sports"]],
              description=desc,
              subprompts=[ner_extract, team_describe],
              code=open("ner.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )

if __name__ == "__main__":
    gradio.queue().launch()
