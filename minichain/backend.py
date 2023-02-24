import os
import subprocess
from dataclasses import dataclass
from types import TracebackType
from typing import List, Optional, Sequence
import json, sys
from eliottree import tasks_from_iterable, render_tasks

from eliot import start_action, to_file


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


x = "text-davinci-003"


class OpenAI(Backend):
    def __init__(self, model: str = x) -> None:

        import openai

        self.api_key = os.environ.get("OPENAI_KEY")
        assert self.api_key, "Need an OPENAI_KEY. Get one here https://openai.com/api/"
        import async_openai
        openai.api_key = self.api_key

        self.options = dict(
            model=model,
            max_tokens=256,
            temperature=0,
        )

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
        async_openai.OpenAI.configure(api_key=self.api_key,
                                      debug_enabled = False,)
        ans = await async_openai.OpenAI.Completions.async_create(
            **self.options,
            stop=request.stop,
            prompt=request.prompt,
        )
        return str(ans.choices[0].text)

    

class HuggingFace(Backend):
    def __init__(self, model: str = "gpt2") -> None:
        self.model = model

    def run(self, request: Request) -> str:
        import hfapi
        client = hfapi.Client()
        x = client.text_generation(request.prompt, model=self.model)
        return x["generated_text"][len(request.prompt):]
    
    
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
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.action.finish()

        
    Mock = Mock
    Google = Google
    OpenAI = OpenAI
    BashProcess = BashProcess
    Python = Python


def start_chain(name: str) -> _MiniChain:
    return _MiniChain(name)

def show_log(s, o=sys.stderr.write):
    render_tasks(o,
                 tasks_from_iterable([json.loads(l) for l in open(s)]),
                 colorize=True, human_readable=True)
