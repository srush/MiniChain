# Prompt from ...
#
from dataclasses import dataclass

from parsita import TextParsers, lit, reg

from minichain import Backend, JinjaPrompt, SimplePrompt, start_chain


# Define the states of the bot.
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

    class Parser(TextParsers):  # type: ignore
        follow = (lit("Follow up:") >> reg(r".*")) > IntermediateState
        finish = (lit("So the final answer is: ") >> reg(r".*")) > FinalState
        response = follow | finish

    def parse(self, response: str, inp: JinjaPrompt.IN) -> Out:
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
        suffix += "\nIntermediate answer:" + out2
    return out.state.s


with start_chain("selfask") as backend:
    result = selfask(
        "What is the zip code of the city where George Washington was born?",
        backend.Mock(
            [
                "Follow up: Where was George Washington born?",
                "Follow up: What is the zip code of Virginia?",
                "So the final answer is: 12312",
            ]
        ),
        # OpenAI("")
        backend.Mock(["Virginia", "12312"]),
    )
print(result)


# Google("593a073fa4c730efe918e592a538b36e80841bc8f8dd4070c1566920f75ba140")))
