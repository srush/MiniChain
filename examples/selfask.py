# Implementation of the self-ask + Google tool use prompt.  
# Adapted from https://github.com/ofirpress/self-ask

from dataclasses import dataclass
from minichain import Backend, JinjaPrompt, SimplePrompt, start_chain, show_log
from parsita import TextParsers, lit, reg


# Define the state of the bot.
@dataclass
class IntermediateState:
    s: str
@dataclass
class FinalState:
    s: str
@dataclass
class Out:
    echo: str
    state: FinalState | IntermediateState


class SelfAsk(JinjaPrompt[Out]):
    template_file = "selfask.pmpt.tpl"
    stop_template = "\nIntermediate answer:"

    # Parsita parser.
    class Parser(TextParsers):
        follow = (lit("Follow up:") >> reg(r".*")) > IntermediateState
        finish = (lit("So the final answer is: ") >> reg(r".*")) > FinalState
        response = follow | finish

    def parse(self, response: str, inp) -> Out:
        return Out(
            self.prompt(inp).prompt + response,
            self.Parser.response.parse(response).or_die(),
        )
SelfAsk().show(
    {
        "input": "What is the zip code of the city where George Washington was born?",
        "agent_scratchpad": True,
    },
    "Follow up: Where was George Washington born?",
)


def selfask(inp: str, openai: Backend, google: Backend) -> str:
    prompt1 = SelfAsk(openai)
    prompt2 = SimplePrompt(google)
    suffix = ""
    for i in range(3):
        out = prompt1(dict(input=inp, suffix=suffix, agent_scratchpad=True))

        if isinstance(out.state, FinalState):
            break
        suffix += out.echo
        out2 = prompt2(out.state.s)
        suffix += "\nIntermediate answer: " + out2 + "\n"
    return out.state.s


with start_chain("selfask") as backend:
    result = selfask(
        "What is the zip code of the city where George Washington was born?",
        backend.OpenAI(),
        backend.Google()
    )
    print(result)


show_log("selfask.log")
