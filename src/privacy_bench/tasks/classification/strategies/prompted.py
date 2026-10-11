"""Ask a chat model for one JSON category object and score the generated text strictly."""

from ....core.conditions import Condition
from ....models.chat_model import CHAT_TEMPLATE_KWARGS, ChatModel
from ....models.hub import download_snapshot, resolve_snapshot, snapshot_identity
from ..prompts import build_prompt, load_prompt
from .settings import model_settings, option_settings, text_sha256


class PromptedStrategy:
    name = "prompted"
    timing_scope = "prompt_processing_and_generation"

    def __init__(self, condition: Condition):
        self.model = model_settings(condition)
        self.options = option_settings(condition, required=("prompt",),
                                       optional={"max_new_tokens": 32, "seed": 42})
        self.prompt_spec = load_prompt(self.options["prompt"])
        self._backend = None

    @property
    def input_spec(self) -> dict:
        return {"name": "prompted", "prompt_spec": self.prompt_spec, "prompt_template": "json-label-id-v1",
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS, "max_new_tokens": self.options["max_new_tokens"],
                "seed": self.options["seed"], "batch_size": 1}

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"])
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
        self._backend = ChatModel(snapshot, precision=self.model["precision"],
                                  max_new_tokens=self.options["max_new_tokens"], seed=self.options["seed"])
        identity = snapshot_identity(self.model["id"], self.model["revision"], snapshot)
        return {**identity, **self._backend.load()}

    def prepare(self, row) -> dict:
        prompt = build_prompt(row.text, self.prompt_spec)
        return {"prompt": prompt, "chat": self._backend.encode(prompt)}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["chat"]["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        # Each call is an independent single-turn conversation; no history carries over.
        return {"prompt_sha256": text_sha256(prepared["prompt"]), **self._backend.generate(prepared["chat"])}
