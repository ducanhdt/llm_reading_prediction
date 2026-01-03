import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_TOKENIZER_CACHE = {}
_BASE_MODEL_CACHE = {}


def get_tokenizer(path):
    if path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[path]


def load_model(cfg):
    """
    cfg examples:
    {type: hf, path: gpt2}
    {type: full, path: /path/to/model}
    {type: peft, base: google/gemma-3-1b-it, path: adapter_dir}
    """

    device_map = "auto"
    dtype = torch.float16 if torch.cuda.is_available() else None

    if cfg["type"] in ["hf", "full"]:
        tok = get_tokenizer(cfg["path"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=dtype,
        )

    elif cfg["type"] == "peft":
        tok = get_tokenizer(cfg["base"])

        if cfg["base"] not in _BASE_MODEL_CACHE:
            _BASE_MODEL_CACHE[cfg["base"]] = AutoModelForCausalLM.from_pretrained(
                cfg["base"],
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=dtype,
            )

        model = PeftModel.from_pretrained(
            _BASE_MODEL_CACHE[cfg["base"]],
            cfg["path"],
        )

    else:
        raise ValueError(cfg)

    model.eval()
    return tok, model