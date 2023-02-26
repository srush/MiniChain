import json
import os
import subprocess
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Sequence

from eliot import start_action, to_file
from eliottree import render_tasks, tasks_from_iterable

if TYPE_CHECKING:
    import manifest


@dataclass
class Request:
    prompt: str
    stop: Optional[Sequence[str]] = None


class Backend:
    def run(self, request: Request) -> str:
        raise NotImplementedError

    async def arun(self, request: Request) -> str:
        return self.run(request)


class Mock(Backend):
    def __init__(self, answers: List[str] = []):
        self.i = 0
        self.answers = answers

    def run(self, request: Request) -> str:
        self.i += 1
        return self.answers[self.i - 1]


class Google(Backend):
    def __init__(self) -> None:
        serpapi_key = os.environ.get("SERP_KEY")
        assert (
            serpapi_key
        ), "Need a SERP_KEY. Get one here https://serpapi.com/users/welcome"
        self.serpapi_key = serpapi_key

    def run(self, request: Request) -> str:
        from serpapi import GoogleSearch

        params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": request.prompt,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }

        search = GoogleSearch(params)
        res = search.get_dict()

        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]
        else:
            toret = ""
        return str(toret)


class Python(Backend):
    """Executes bash commands and returns the output."""

    def run(self, request: Request) -> str:
        """Run commands and return final output."""
        from contextlib import redirect_stdout
        from io import StringIO

        f = StringIO()
        with redirect_stdout(f):
            exec(request.prompt)
        s = f.getvalue()
        return s


class BashProcess(Backend):
    """Executes bash commands and returns the output."""

    def __init__(self, strip_newlines: bool = False, return_err_output: bool = False):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output

    def run(self, request: Request) -> str:
        """Run commands and return final output."""
        try:
            output = subprocess.run(
                request.prompt,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.decode()
        except subprocess.CalledProcessError as error:
            if self.return_err_output:
                return str(error.stdout.decode())
            return str(error)
        if self.strip_newlines:
            output = output.strip()
        return output



class OpenAIBase(Backend):
    def __init__(self, model: str = "text-davinci-003") -> None:

        import openai

        self.api_key = os.environ.get("OPENAI_KEY")
        assert self.api_key, "Need an OPENAI_KEY. Get one here https://openai.com/api/"

        openai.api_key = self.api_key
        self.model = model
        self.options = dict(
            model=model,
            max_tokens=256,
            temperature=0,
        )
    
class OpenAI(OpenAIBase):
    def run(self, request: Request) -> str:
        import openai

        ans = openai.Completion.create(
            **self.options,
            stop=request.stop,
            prompt=request.prompt,
        )
        return str(ans["choices"][0]["text"])

    async def arun(self, request: Request) -> str:
        import async_openai

        async_openai.OpenAI.configure(
            api_key=self.api_key,
            debug_enabled=False,
        )
        ans = await async_openai.OpenAI.Completions.async_create(
            **self.options,
            stop=request.stop,
            prompt=request.prompt,
        )
        return str(ans.choices[0].text)

class OpenAIEmbed(OpenAIBase):
    def __init__(self, model = "text-embedding-ada-002"):
        super().__init__(model)
        
    def run(self, request: Request) -> str:
        import openai

        ans = openai.Embedding.create(
            engine = self.model,
            input=request.prompt,
        )
        return ans["data"][0]["embedding"]


class HuggingFace(Backend):
    def __init__(self, model: str = "gpt2") -> None:
        self.model = model

    def run(self, request: Request) -> str:
        import hfapi

        client = hfapi.Client()
        x = client.text_generation(request.prompt, model=self.model)
        return x["generated_text"][len(request.prompt) :]  # type: ignore


class Manifest(Backend):
    def __init__(self, client: "manifest.Manifest") -> None:
        self.client = client

    def run(self, request: Request) -> str:
        try:
            import manifest
        except ImportError:
            raise ImportError(
                "`pip install manifest-ml` to use the Manifest Backend."
            )
        assert isinstance(self.client, manifest.Manifest), \
            "Client must be a `manifest.Manifest` instance."

        return self.client.run(request.prompt)


class _MiniChain:
    def __init__(self, name: str):
        to_file(open(f"{name}.log", "w"))
        self.name = name

    def __enter__(self) -> "_MiniChain":
        self.action = start_action(action_type=self.name)
        return self

    def __exit__(
        self,
        type: type[BaseException],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.action.finish()

    Mock = Mock
    Google = Google
    OpenAI = OpenAI
    OpenAIEmbed = OpenAIEmbed
    BashProcess = BashProcess
    Python = Python
    Manifest = Manifest


def start_chain(name: str) -> _MiniChain:
    return _MiniChain(name)


def show_log(s: str, o=sys.stderr.write) -> None:
    render_tasks(
        o,
        tasks_from_iterable([json.loads(l) for l in open(s)]),
        colorize=True,
        human_readable=True,
    )
