# project/src/models_chat.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ───────────────────────────────────────────────────────────────────────
# Helper: detect best dtype
# ───────────────────────────────────────────────────────────────────────

def get_best_dtype():
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


# ───────────────────────────────────────────────────────────────────────
# Chat Model Wrapper
# ───────────────────────────────────────────────────────────────────────

@dataclass
class HFChatModel:
    model_id: str
    load_in_4bit: bool = True
    template_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):

        print(f"[HFChatModel] Loading tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": get_best_dtype()
        }

        # Optional 4-bit loading via BitsAndBytesConfig
        if self.load_in_4bit:
            try:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs.pop("torch_dtype", None)
                print("[HFChatModel] Using 4-bit quantization")
            except Exception:
                print("[HFChatModel] 4-bit unavailable, falling back")

        print(f"[HFChatModel] Loading model: {self.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        self.model.eval()

    # ───────────────────────────────────────────────────────────────────

    def _render_with_template(self, messages: List[Dict[str, str]]) -> str:

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.template_kwargs
            )
        except Exception:
            return self._fallback_render(messages)

    # ───────────────────────────────────────────────────────────────────

    def _fallback_render(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback for models without chat templates.
        Simple role-based formatting.
        """

        parts = []

        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            parts.append(f"{role}:\n{content}\n")

        parts.append("ASSISTANT:\n")

        return "\n".join(parts)

    # ───────────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_new_tokens: int
    ) -> Tuple[str, int, int, bool]:

        prompt = self._render_with_template(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 0.0

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if do_sample:
            gen_kwargs.update({
                "temperature": float(temperature),
                "top_p": float(top_p)
            })

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        input_len = inputs["input_ids"].shape[1]

        gen_tokens = outputs[0][input_len:]

        text = self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        ).strip()

        hit_max = len(gen_tokens) >= max_new_tokens

        return (
            text,
            int(input_len),
            int(len(gen_tokens)),
            bool(hit_max)
        )

    # ───────────────────────────────────────────────────────────────────

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
