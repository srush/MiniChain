import os
import subprocess
import time
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from eliot import start_action, to_file

if TYPE_CHECKING:
    import manifest


class Backend:
    @property
    def description(self) -> str:
        return ""

    def run(self, request: str) -> str:
        raise NotImplementedError

    def run_stream(self, request: str) -> Iterator[str]:
        yield self.run(request)

    async def arun(self, request: str) -> str:
        return self.run(request)

    def _block_input(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)

    def _block_output(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)


class Id(Backend):
    def run(self, request: str) -> str:
        return request


class Mock(Backend):
    def __init__(self, answers: List[str] = []):
        self.i = -1
        self.answers = answers

    def run(self, request: str) -> str:
        self.i += 1
        return self.answers[self.i % len(self.answers)]

    def run_stream(self, request: str) -> Iterator[str]:
        self.i += 1
        result = self.answers[self.i % len(self.answers)]
        for c in result:
            yield c
            time.sleep(0.1)

    def __repr__(self) -> str:
        return f"Mocked Backend {self.answers}"


class Google(Backend):
    def __init__(self) -> None:
        pass

    def run(self, request: str) -> str:
        from serpapi import GoogleSearch

        serpapi_key = os.environ.get("SERP_KEY")
        assert (
            serpapi_key
        ), "Need a SERP_KEY. Get one here https://serpapi.com/users/welcome"
        self.serpapi_key = serpapi_key

        params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": request,
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

    def __repr__(self) -> str:
        return "Google Search Backend"


class Python(Backend):
    """Executes Python commands and returns the output."""

    def _block_input(self, gr):  # type: ignore
        return gr.Code()

    def _block_output(self, gr):  # type: ignore
        return gr.Code()

    def run(self, request: str) -> str:
        """Run commands and return final output."""
        from contextlib import redirect_stdout
        from io import StringIO

        p = request.strip()
        if p.startswith("```"):
            p = "\n".join(p.strip().split("\n")[1:-1])

        f = StringIO()
        with redirect_stdout(f):
            exec(p)
        s = f.getvalue()
        return s

    def __repr__(self) -> str:
        return "Python-Backend"


class Bash(Backend):
    """Executes bash commands and returns the output."""

    def _block_input(self, gr):  # type: ignore
        return gr.Code()

    def _block_output(self, gr):  # type: ignore
        return gr.Code()

    def __init__(self, strip_newlines: bool = False, return_err_output: bool = False):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output

    def run(self, request: str) -> str:
        """Run commands and return final output."""
        try:
            output = subprocess.run(
                request,
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

    def __repr__(self) -> str:
        return "Bash-Backend"


class OpenAIBase(Backend):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.stop = stop
        self.options = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"OpenAI Backend {self.options}"


class OpenAI(OpenAIBase):
    def run(self, request: str) -> str:
        import manifest

        chat = {"gpt-4", "gpt-3.5-turbo"}
        manifest = manifest.Manifest(
            client_name="openaichat" if self.model in chat else "openai",
            max_tokens=self.options["max_tokens"],
            cache_name="sqlite",
            cache_connection=f"{MinichainContext.name}",
        )

        ans = manifest.run(
            request,
            stop_sequences=self.stop,
        )
        return str(ans)

    def run_stream(self, prompt: str) -> Iterator[str]:
        import openai

        self.api_key = os.environ.get("OPENAI_API_KEY")
        assert (
            self.api_key
        ), "Need an OPENAI_API_KEY. Get one here https://openai.com/api/"
        openai.api_key = self.api_key

        for chunk in openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stop=self.stop,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                yield content


class OpenAIEmbed(OpenAIBase):
    def _block_output(self, gr):  # type: ignore
        return gr.Textbox(label="Embedding")

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

    def run(self, request: str) -> str:
        import openai

        self.api_key = os.environ.get("OPENAI_API_KEY")
        assert (
            self.api_key
        ), "Need an OPENAI_API_KEY. Get one here https://openai.com/api/"
        openai.api_key = self.api_key

        ans = openai.Embedding.create(
            engine=self.model,
            input=request,
        )
        return ans["data"][0]["embedding"]  # type: ignore


class HuggingFaceBase(Backend):
    def __init__(self, model: str = "gpt2") -> None:
        self.model = model


class HuggingFace(HuggingFaceBase):
    def run(self, request: str) -> str:

        from huggingface_hub.inference_api import InferenceApi

        self.api_key = os.environ.get("HF_KEY")
        assert self.api_key, "Need an HF_KEY. Get one here https://huggingface.co/"

        self.client = InferenceApi(
            token=self.api_key, repo_id=self.model, task="text-generation"
        )
        response = self.client(inputs=request)
        return response  # type: ignore


class HuggingFaceEmbed(HuggingFaceBase):
    def run(self, request: str) -> str:

        from huggingface_hub.inference_api import InferenceApi

        self.api_key = os.environ.get("HF_KEY")
        assert self.api_key, "Need an HF_KEY. Get one here https://huggingface.co/"

        self.client = InferenceApi(
            token=self.api_key, repo_id=self.model, task="feature-extraction"
        )
        response = self.client(inputs=request)
        return response  # type: ignore


class Manifest(Backend):
    def __init__(self, client: "manifest.Manifest") -> None:
        "Client from [Manifest-ML](https://github.com/HazyResearch/manifest)."
        self.client = client

    def run(self, request: str) -> str:
        try:
            import manifest
        except ImportError:
            raise ImportError("`pip install manifest-ml` to use the Manifest Backend.")
        assert isinstance(
            self.client, manifest.Manifest
        ), "Client must be a `manifest.Manifest` instance."

        return self.client.run(request)  # type: ignore


@dataclass
class RunLog:
    request: str = ""
    response: Optional[str] = ""
    output: str = ""
    dynamic: int = 0


@dataclass
class PromptSnap:
    input_: Any = ""
    run_log: RunLog = RunLog()
    output: Any = ""


class MinichainContext:
    id_: int = 0
    prompt_store: Dict[Tuple[int, int], List[PromptSnap]] = {}
    prompt_count: Dict[int, int] = {}
    name: str = ""


def set_minichain_log(name: str) -> None:
    to_file(open(f"{name}.log", "w"))


class MiniChain:
    """
    MiniChain session object with backends. Make backend by calling
    `minichain.OpenAI()` with args for `OpenAI` class.
    """

    def __init__(self, name: str):
        to_file(open(f"{name}.log", "w"))
        self.name = name

    def __enter__(self) -> "MiniChain":
        MinichainContext.prompt_store = {}
        MinichainContext.prompt_count = {}
        MinichainContext.name = self.name
        self.action = start_action(action_type=self.name)
        return self

    def __exit__(
        self,
        type: type,
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.action.finish()
        self.prompt_store = dict(MinichainContext.prompt_store)
        MinichainContext.prompt_store = {}
        MinichainContext.prompt_count = {}
        MinichainContext.name = ""


def start_chain(name: str) -> MiniChain:
    """
    Initialize a chain. Logs to {name}.log. Returns a `MiniChain` that
    holds LLM backends..
    """
    return MiniChain(name)


# def show_log(filename: str, o: Callable[[str], Any] = sys.stderr.write) -> None:
#     """
#     Write out the full asynchronous log from file `filename`.
#     """
#     render_tasks(
#         o,
#         tasks_from_iterable([json.loads(line) for line in open(filename)]),
#         colorize=True,
#         human_readable=True,
#     )
