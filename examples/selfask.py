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


States = FinalState | IntermediateState


class SelfAsk(JinjaPrompt[States]):
    template_file = "selfask.pmpt.tpl"
    stop_template = "\nIntermediate answer:"

    class SelfAskParser(TextParsers):  # type: ignore
        follow = (lit("Follow up:") >> reg(r".*")) > IntermediateState
        finish = (lit("So the final answer is: ") >> reg(r".*")) > FinalState
        response = follow | finish

    def parse(self, inp: str) -> States:
        return self.SelfAskParser.response.parse(inp).or_die()  # type: ignore


def selfask(inp: str, openai: Backend, google: Backend) -> str:
    prompt1 = SelfAsk(openai)
    prompt2 = SimplePrompt(google)
    suffix = ""
    for i in range(3):
        r1 = prompt1(
            dict(input=inp, suffix=suffix, agent_scratchpad=True), name=f"Chat {i}"
        )
        if isinstance(r1.val, FinalState):
            break
        out = prompt2(r1.val.s, name=f"Google{i}")
        suffix += "\nIntermediate answer:" + out.echo
    return r1.val.s


if __name__ == "__main__":
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
