import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
from peft import PeftModel

# =========================================================
# Utilities
# =========================================================

def _find_ckpt_dir(model_dir):
    """
    model_dir/
        └── model_step_XXXX/
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(model_dir)

    for name in os.listdir(model_dir):
        sub = os.path.join(model_dir, name)
        if os.path.isdir(sub) and name.startswith("model_step"):
            return sub

    raise FileNotFoundError(f"No model_step_xxxx found in {model_dir}")


def _resolve_gpt2_local_path(model_name):
    """
    Resolve GPT-2 local finetuned checkpoints
    """
    base_root = "/kaggle/working/gpt2_models"
    small_root = os.path.join(base_root, "small")
    medium_root = os.path.join(base_root, "medium")

    # absolute ckpt path
    if os.path.isabs(model_name) and os.path.isdir(model_name):
        tokenizer_root = os.path.dirname(os.path.dirname(model_name))
        return model_name, tokenizer_root

    # baseline
    if model_name in ["gpt2", "gpt2-medium", "gpt2_medium"]:
        return model_name, None

    # finetuned small
    small_ft = ["all_small_fd_config_8", "all_small_ffd_config_8"]
    if model_name in small_ft:
        model_dir = os.path.join(small_root, model_name)
        ckpt = _find_ckpt_dir(model_dir)
        return ckpt, small_root

    # finetuned medium
    medium_ft = ["all_medium_fd_0_kl_config_42", "all_medium_ffd_0_kl_config_42"]
    if model_name in medium_ft:
        model_dir = os.path.join(medium_root, model_name)
        ckpt = _find_ckpt_dir(model_dir)
        return ckpt, medium_root

    return None, None


def _infer_base_model_from_adapter(adapter_path: str):
    name = adapter_path.lower()
    if "gemma" in name:
        return "google/gemma-3-1b-it"
    if "qwen" in name:
        return "Qwen/Qwen3-0.6B"
    raise ValueError(f"Cannot infer base model from adapter path: {adapter_path}")


# =========================================================
# Model Loaders
# =========================================================

class ModelLoaderCausal:
    """
    Unified causal LM loader:
      - GPT-2 baseline (HF)
      - GPT-2 local finetune
      - Gemma / Qwen + LoRA adapter
    """

    def load_tokenizer(self, model_name: str):
        # ---- LoRA adapters (Gemma / Qwen) ----
        if "gemma" in model_name.lower() or "qwen" in model_name.lower():
            base = _infer_base_model_from_adapter(model_name)
            print(f"[Tokenizer] HF base → {base}")
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            tok.padding_side = "left"
            tok.truncation_side = "left"
            return tok

        # ---- GPT-2 family ----
        model_path, tokenizer_path = _resolve_gpt2_local_path(model_name)

        if tokenizer_path is None:
            print(f"[Tokenizer] HF baseline → {model_name}")
            return AutoTokenizer.from_pretrained(model_name, use_fast=True)

        print(f"[Tokenizer] local → {tokenizer_path}")
        return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


    def load_model(self, model_name: str):
        # ---- LoRA adapters (Gemma / Qwen) ----
        if "gemma" in model_name.lower() or "qwen" in model_name.lower():
            adapter_path = model_name
            base_model = _infer_base_model_from_adapter(adapter_path)

            print(f"[Model] base → {base_model}")
            print(f"[Model] adapter → {adapter_path}")

            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                output_attentions=True,
            )

            model = PeftModel.from_pretrained(base, adapter_path)
            model.eval()
            return model

        # ---- GPT-2 family ----
        model_path, tokenizer_path = _resolve_gpt2_local_path(model_name)

        if tokenizer_path is None:
            print(f"[Model] HF baseline → {model_name}")
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
            )

        print(f"[Model] local → {model_path}")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            output_attentions=True,
        )


# =========================================================
# Reward / RM / Others (unchanged)
# =========================================================

class ModelLoaderReward:
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def load_model(self, model_name):
        print("[ModelLoaderReward]")
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
        )


class ModelLoaderUltra:
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def load_model(self, model_name):
        print("[ModelLoaderUltra]")
        return LlamaRewardModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_attentions=True,
        )


class ModelLoaderQRLlama:
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    def load_model(self, model_name):
        print("[ModelLoaderQRLlama]")
        return LlamaForRewardModelWithGating.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
        )


class ModelLoaderEurus:
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def load_model(self, model_name):
        print("[ModelLoaderEurus]")
        return EurusRewardModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_attentions=True,
        )


class ModelLoaderBert:
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def load_model(self, model_name):
        return AutoModelForMaskedLM.from_pretrained(
            model_name,
            output_attentions=True,
        )


# =========================================================
# Factory
# =========================================================

class ModelLoaderFactory:
    def get_model_loader(self, loader_type: str):
        if loader_type == "BertBased":
            return ModelLoaderBert()
        if loader_type == "causalLM":
            return ModelLoaderCausal()
        if loader_type == "reward":
            return ModelLoaderReward()
        if loader_type == "ultraRM":
            return ModelLoaderUltra()
        if loader_type == "QRLlama":
            return ModelLoaderQRLlama()
        if loader_type == "eurus":
            return ModelLoaderEurus()

        raise ValueError(f"Unknown model loader type: {loader_type}")