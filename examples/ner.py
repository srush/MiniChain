# + tags=["hide_inp"]

desc = """
### Named Entity Recognition

Chain that does named entity recognition with arbitrary labels. [[Code](https://github.com/srush/MiniChain/blob/main/examples/ner.py)]

(Adapted from [promptify](https://github.com/promptslab/Promptify/blob/main/promptify/prompts/nlp/templates/ner.jinja)).
"""
# -

# $

from minichain import prompt, show, OpenAI

@prompt(OpenAI(), template_file = "ner.pmpt.tpl", parser="json")
def ner_extract(model, **kwargs):
    return model(kwargs)

@prompt(OpenAI())
def team_describe(model, inp):
    query = "Can you describe these basketball teams? " + \
        " ".join([i["E"] for i in inp if i["T"] =="Team"])
    return model(query)


def ner(text_input, labels, domain):
    extract = ner_extract(dict(text_input=text_input, labels=labels, domain=domain))
    return team_describe(extract)


# $

gradio = show(ner,
              examples=[["An NBA playoff pairing a year ago, the 76ers (39-20) meet the Miami Heat (32-29) for the first time this season on Monday night at home.", "Team, Date", "Sports"]],
              description=desc,
              subprompts=[ner_extract, team_describe],
              code=open("ner.py", "r").read().split("$")[1].strip().strip("#").strip(),
              )

if __name__ == "__main__":
    gradio.launch()


# View prompt examples.

# + tags=["hide_inp"]
# NERPrompt().show(
#     {
#         "input": "I went to New York",
#         "domain": "Travel",
#         "labels": ["City"]
#     },
#     '[{"T": "City", "E": "New York"}]',
# )
# # -

# # View log.

# minichain.show_log("ner.log")
