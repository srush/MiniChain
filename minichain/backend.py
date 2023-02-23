import subprocess
from types import TracebackType
from typing import List, Sequence

from eliot import start_action, to_file


class Backend:
    def run(self, question: str, stop: Sequence[str]) -> str:
        pass


class Mock(Backend):
    def __init__(self, answers: List[str] = []):
        self.i = 0
        self.answers = answers

    def run(self, question: str, stop: Sequence[str]) -> str:
        self.i += 1
        return self.answers[self.i - 1]


class Google(Backend):
    def __init__(self, serpapi_key: str):

        self.serpapi_key = serpapi_key

    def run(self, question: str, stop: Sequence[str]) -> str:
        from serpapi import GoogleSearch

        params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": question,
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


class BashProcess(Backend):
    """Executes bash commands and returns the output."""

    def __init__(self, strip_newlines: bool = False, return_err_output: bool = False):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output

    def run(self, commands: str, stop: Sequence[str]) -> str:
        """Run commands and return final output."""
        # if isinstance(commands, str):
        #     commands = [commands]
        # commands = ";".join(commands)
        try:
            output = subprocess.run(
                commands,
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


class OpenAI(Backend):
    def __init__(self, api_key: str):
        import openai

        openai.api_key = api_key

    def run(self, question: str, stop: Sequence[str]) -> str:
        import openai

        ans = openai.Completion.create(
            model="text-davinci-003",
            max_tokens=256,
            # stop=stop,
            prompt=question,
            temperature=0,
        )
        return str(ans["choices"][0]["text"])


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


def start_chain(name: str) -> _MiniChain:
    return _MiniChain(name)
