import trio
from eliot import start_action

from minichain import SimplePrompt, start_chain


async def promptee(mock):
    out = mock(SimplePrompt, input="b", name=f"F1")
    async with trio.open_nursery() as nursery:
        nursery.start_soon(mock.ask, SimplePrompt, dict(input="b", name=f"F1"))
        nursery.start_soon(mock.ask, SimplePrompt, dict(input="a", name=f"F2"))


with start_chain("parallel") as backend:
    trio.run(promptee, backend.Mock(["a", "b", "b", "d"]))
