"""Ask a chat model to rewrite one passage, with the system message and template from a prompt file."""

from ...core.conditions import Condition, take_settings
from ...core.files import digest_text
from ...models.chat_model import CHAT_TEMPLATE_KWARGS, ChatModel
from ...models.gguf_chat_model import GgufChatModel
from ...models.hub import download_snapshot, require_commit, resolve_snapshot, snapshot_identity
from .data import load_prompt, render


class PromptedRewriteStrategy:
    name = "prompted"
    timing_scope = "prompt_processing_and_generation"
    timing_description = "prompt processing + generation"
    generates_text = True

    @staticmethod
    def condition_label(spec: dict) -> str:
        return spec["prompt"]["name"]

    def __init__(self, condition: Condition):
        context = f"Condition {condition.name!r} [model]"
        if "file" in condition.model:
            # A GGUF file runs through llama.cpp, which has no compute type and no default device.
            self.model = take_settings(condition.model, context,
                                       required=("id", "revision", "precision", "file", "device"))
        else:
            self.model = take_settings(condition.model, context, required=("id", "revision", "precision"),
                                       optional={"device": "cuda", "compute": "bf16"})
        require_commit(self.model["id"], self.model["revision"])
        self.options = take_settings(condition.options, f"Condition {condition.name!r} [options]",
                                     required=("prompt",), optional={"max_new_tokens": 1024, "seed": 42})
        self.prompt = load_prompt(self.options["prompt"])
        self._backend = None

    @property
    def input_spec(self) -> dict:
        return {"name": "prompted-rewrite", "prompt": self.prompt, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "max_new_tokens": self.options["max_new_tokens"], "seed": self.options["seed"], "batch_size": 1}

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"], only=self.model.get("file"))
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        settings = {"precision": self.model["precision"], "device": self.model["device"],
                    "max_new_tokens": self.options["max_new_tokens"], "seed": self.options["seed"]}
        if "file" in self.model:
            snapshot = resolve_snapshot(self.model["id"], self.model["revision"], self.model["file"])
            self._backend = GgufChatModel(snapshot / self.model["file"], **settings)
        else:
            snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
            self._backend = ChatModel(snapshot, compute=self.model["compute"], **settings)
        identity = snapshot_identity(self.model["id"], self.model["revision"], snapshot)
        return {**identity, **self._backend.load()}

    def prepare(self, passage) -> dict:
        messages = render(self.prompt, passage)
        return {"messages": messages, "chat": self._backend.encode(messages["user"], system=messages["system"])}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["chat"]["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        # Each call is an independent conversation; no history carries over.
        return {"prompt_sha256": digest_text(prepared["messages"]["user"]), **self._backend.generate(prepared["chat"])}
