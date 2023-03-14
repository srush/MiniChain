# + tags=["hide_inp"]

desc = """
### Named Entity Recognition

Chain that does named entity recognition with arbitrary labels. [[Code](https://github.com/srush/MiniChain/blob/main/examples/ner.py)]

(Adapted from [promptify](https://github.com/promptslab/Promptify/blob/main/promptify/prompts/nlp/templates/ner.jinja)).
"""
# -

import json

import minichain

# Prompt to extract NER tags as json

class NERPrompt(minichain.TemplatePrompt):
    template_file = "ner.pmpt.tpl"

    def parse(self, response, inp):
        return json.loads(response)

# Use NER to ask a simple queston.

class TeamPrompt(minichain.Prompt):
    def prompt(self, inp):
        return "Can you describe these basketball teams? " + \
            " ".join([i["E"] for i in inp if i["T"] =="Team"])

    def parse(self, response, inp):
        return response

# Run the system.

with minichain.start_chain("ner") as backend:
    ner_prompt = NERPrompt(backend.OpenAI())
    team_prompt = TeamPrompt(backend.OpenAI())
    prompt = ner_prompt.chain(team_prompt)
    # results = prompt(
    #     {"text_input": "An NBA playoff pairing a year ago, the 76ers (39-20) meet the Miami Heat (32-29) for the first time this season on Monday night at home.",
    #      "labels" : ["Team", "Date"],
    #      "domain": "Sports"
    #      }
    # )
    # print(results)

gradio = prompt.to_gradio(fields =["text_input", "labels", "domain"],
                          examples=[["An NBA playoff pairing a year ago, the 76ers (39-20) meet the Miami Heat (32-29) for the first time this season on Monday night at home.", "Team, Date", "Sports"]],
                          description=desc)

    
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
