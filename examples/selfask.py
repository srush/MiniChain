# Prompt from ...
#
from dataclasses import dataclass
from typing import Mapping

from parsita import TextParsers, lit, reg

from minichain import Backend, JinjaPrompt, SimplePrompt, start_chain


@dataclass
class Intermediate:
    s: str


@dataclass
class Final:
    s: str


class SelfAsk(JinjaPrompt[Final | Intermediate]):
    prompt_template = self_ask_prompt = """
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No

Question: {{input}}
Are followup questions needed here: {% if agent_scratchpad %}Yes{%else%}No{% endif %}.
{{suffix}}
"""

    class SelfAskParser(TextParsers):  # type: ignore
        follow = (lit("Follow up:") >> reg(r".*")) > Intermediate
        finish = (lit("So the final answer is: ") >> reg(r".*")) > Final
        response = follow | finish

    @classmethod
    def parse(cls, inp: str) -> Intermediate | Final:
        return cls.SelfAskParser.response.parse(inp).or_die()  # type: ignore

    @classmethod
    def stop(cls, kwargs: Mapping[str, str]) -> str:
        return "\nIntermediate answer:"


def selfask(inp: str, openai: Backend, google: Backend) -> str:
    suffix = ""
    for i in range(3):
        r1 = SelfAsk.run(
            openai, dict(input=inp, suffix=suffix, agent_scratchpad=True), f"Chat {i}"
        )
        if isinstance(r1.val, Final):
            break
        out = SimplePrompt.run(google, r1.val.s, f"Google{i}")
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
            # OpenAI("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"),
            backend.Mock(["Virginia", "12312"]),
        )
    print(result)
    # Google("593a073fa4c730efe918e592a538b36e80841bc8f8dd4070c1566920f75ba140")))
