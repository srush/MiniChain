# Notebook implementation of named entity recognition.
# Adapted from [promptify](https://github.com/promptslab/Promptify/blob/main/promptify/prompts/nlp/templates/ner.jinja).


import minichain
import json

# Prompt to extract NER tags as json

class NERPrompt(minichain.TemplatePrompt):
    template_file = "ner.pmpt.tpl"
    
    def parse(self, response, inp):
        return json.loads(response)


# Run the system.

with minichain.start_chain("ner") as backend:
    p1 = NERPrompt(backend.OpenAI())
    results = p1(
        {"text_input": "An NBA playoff pairing a year ago, the 76ers (39-20) meet the Miami Heat (32-29) for the first time this season on Monday night at home.",
         "labels" : ["Team", "Date"],
         "domain": "Sports"
         }
    )
    print(results)

# View prompt examples.

# + tags=["hide_inp"]
NERPrompt().show(
    {
        "input": "I went to New York",
        "domain": "Travel",
        "labels": ["City"]
    },
    '[{"T": "City", "E": "New York"}]',
)
# -

# View log.

minichain.show_log("ner.log")
