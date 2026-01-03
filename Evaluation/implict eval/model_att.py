import os
import numpy as np
import pandas as pd
import torch
import scipy.special

from transformers import BatchEncoding

from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from models.model_loader import ModelLoaderFactory


class ModelAttentionExtractor:
    """
    Universal attention extractor for decoder-only LMs (GPT-2 / Gemma / Qwen)
    and (best-effort) encoder models.

    Key choices:
      - force output_attentions=True and return_dict=True
      - robust special token id collection
      - word-level mapping via FixationsAligner + TokenizerAligner fallback
    """

    def __init__(self, model_name: str, model_type: str):
        """
        model_type: must match your ModelLoaderFactory routing
                   e.g. "GPT2", "Gemma", "Qwen", "BertBased", "QRLlama", "ultraRM", "eurus", ...
        """
        self.model_type = model_type
        model_loader = ModelLoaderFactory().get_model_loader(model_type)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_loader.load_model(model_name).to(self.device)
        self.model.eval()

        self.tokenizer = model_loader.load_tokenizer(model_name)

        # chat template info (if exists)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            print("tokenizer chat template:")
            print(self.tokenizer.chat_template)
        else:
            print("tokenizer has no chat template (likely base LM / GPT2 style)")

        # robust special token ids
        self.special_tokens_id = self._collect_special_token_ids(self.tokenizer)

    @staticmethod
    def _collect_special_token_ids(tokenizer):
        """
        Collect special token ids robustly across tokenizers.
        Preference:
          1) tokenizer.all_special_ids (most reliable)
          2) tokenizer.special_tokens_map (strings or list of strings)
        """
        ids = set()

        # (1) all_special_ids
        if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids is not None:
            for tid in tokenizer.all_special_ids:
                if tid is not None and tid >= 0:
                    ids.add(int(tid))

        # (2) special_tokens_map
        if hasattr(tokenizer, "special_tokens_map") and tokenizer.special_tokens_map is not None:
            for _, tok in tokenizer.special_tokens_map.items():
                if tok is None:
                    continue
                if isinstance(tok, list):
                    for t in tok:
                        tid = tokenizer.convert_tokens_to_ids(t)
                        if tid is not None and tid >= 0:
                            ids.add(int(tid))
                else:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and tid >= 0:
                        ids.add(int(tid))

        return sorted(list(ids))

    # ---------------------------
    # Mapping helpers
    # ---------------------------
    @staticmethod
    def map_attention_from_tokens_to_words_v2(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        list_words_second = TokenizerAligner().tokens_to_words(text, text_tokenized)
        mapped_words_idxs, _ = TokenizerAligner().map_words(list_words_first, list_words_second)

        features_mapped_first_words = TokenizerAligner().map_features_between_paired_list(
            features_mapped_second_words,
            mapped_words_idxs,
            list_words_first,
            mode=mode,
        )
        return features_mapped_first_words

    @staticmethod
    def map_attention_from_tokens_to_words_reward(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        index_init: int = 0,
        mode: str = "mean",
    ):
        list_words_second = TokenizerAligner().tokens_to_words(text, text_tokenized)

        # keep only response part
        list_words_second = list_words_second[index_init + 1 :]
        features_mapped_second_words = features_mapped_second_words[index_init + 1 :]

        mapped_words_idxs, _ = TokenizerAligner().map_words(list_words_first, list_words_second)

        features_mapped_first_words = TokenizerAligner().map_features_between_paired_list(
            features_mapped_second_words,
            mapped_words_idxs,
            list_words_first,
            mode=mode,
        )
        return features_mapped_first_words

    @staticmethod
    def map_attention_from_words_to_words(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        list_words_second = TokenizerAligner().text_to_words(
            text, text_tokenized, text_tokenized.word_ids()
        )

        mapped_words_idxs, _ = TokenizerAligner().map_words(list_words_first, list_words_second)

        features_mapped_first_words = TokenizerAligner().map_features_between_paired_list(
            features_mapped_second_words,
            mapped_words_idxs,
            list_words_first,
            mode="mean",
        )
        return features_mapped_first_words

    # ---------------------------
    # Attention extraction core
    # ---------------------------
    @staticmethod
    def get_attention_model(model, inputs: BatchEncoding):
        """
        Universal forward that tries to force attentions on.
        """
        # move to device (some HF models don't expose .device)
        if hasattr(model, "device"):
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
        else:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

        # Some tokenizers may not provide attention_mask for certain configs; handle gracefully
        kwargs = dict(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        out = model(**kwargs)

        # HF convention: out.attentions exists when output_attentions=True
        return out.attentions

    def process_attention_reward(
        self,
        attention,
        input_ids: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        index_init: int = 0,
        method="sum",
    ):
        attention_layer = {}
        special_token_idx = self.compute_special_token_idx(
            input_ids["input_ids"][0].tolist(), self.special_tokens_id
        )

        for layer in range(len(attention)):
            att_layer = attention[layer][0].detach().float().cpu().numpy()  # [heads, seq, seq]
            mean_attention = np.mean(att_layer, axis=0)  # [seq, seq]

            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)  # [seq]
            else:
                if self.model_type == "BertBased":
                    aggregated_attention = np.mean(mean_attention, axis=0)
                else:
                    aggregated_attention = self.compute_mean_diagonalbewlow(mean_attention)

            # zero out special tokens
            aggregated_attention = [
                0.0 if i in special_token_idx else float(aggregated_attention[i])
                for i in range(len(aggregated_attention))
            ]

            aggregated_attention = self.map_attention_from_tokens_to_words_reward(
                list_word_original,
                text,
                input_ids,
                features_mapped_second_words=aggregated_attention,
                index_init=index_init,
                mode="mean",
            )

            attention_layer[layer] = scipy.special.softmax(aggregated_attention)

        return attention_layer

    def process_attention(
        self,
        attention,
        input_ids: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        word_level: bool = True,
        method="sum",
    ):
        attention_layer = {}
        special_token_idx = self.compute_special_token_idx(
            input_ids["input_ids"][0].tolist(), self.special_tokens_id
        )

        for layer in range(len(attention)):
            att_layer = attention[layer][0].detach().float().cpu().numpy()  # [heads, seq, seq]
            mean_attention = np.mean(att_layer, axis=0)  # [seq, seq]

            # For each token, sum attention received from all tokens
            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)  # [seq]
            else:
                if self.model_type == "BertBased":
                    aggregated_attention = np.mean(mean_attention, axis=0)
                else:
                    aggregated_attention = self.compute_mean_diagonalbewlow(mean_attention)

            if word_level:
                if list_word_original is None:
                    raise ValueError("list_word_original must be provided when word_level=True")
                if text is None:
                    raise ValueError("text must be provided when word_level=True")

                # zero out special tokens
                aggregated_attention = [
                    0.0 if i in special_token_idx else float(aggregated_attention[i])
                    for i in range(len(aggregated_attention))
                ]

                # 1) try FixationsAligner native token->word
                aggregated_attention_mapped_words = FixationsAligner().map_features_from_tokens_to_words(
                    aggregated_attention, input_ids, mode="sum"
                )

                # 2) fallback: TokenizerAligner mapping
                if aggregated_attention_mapped_words is None:
                    aggregated_attention = self.map_attention_from_tokens_to_words_v2(
                        list_word_original,
                        text,
                        input_ids,
                        features_mapped_second_words=aggregated_attention,
                        mode="mean",
                    )
                else:
                    aggregated_attention = self.map_attention_from_words_to_words(
                        list_word_original,
                        text,
                        input_ids,
                        aggregated_attention_mapped_words,
                        mode="mean",
                    )
            else:
                # token-level: remove special tokens
                aggregated_attention = np.delete(np.array(aggregated_attention), special_token_idx)

            attention_layer[layer] = scipy.special.softmax(aggregated_attention)

        return attention_layer

    def extract_attention(self, texts_trials: dict, word_level: bool = True):
        attention_trials = {}
        for trial, list_text in texts_trials.items():
            print("trial", trial)

            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]

            input_ids = self.tokenize_text(self.tokenizer, text)
            attention = self.get_attention_model(self.model, input_ids)

            try:
                attention_trials[trial] = self.process_attention(
                    attention,
                    input_ids,
                    text=text,
                    list_word_original=list_word_original,
                    word_level=word_level,
                )
            except Exception as e:
                print(trial, "error:", e)

        return attention_trials

    # ---------------------------
    # Reward/chat split support
    # ---------------------------
    @staticmethod
    def tokenize_text_chat(tokenizer, prompt, response, model_type="ULTRA"):
        """
        You can extend this for your different chat formats.
        """
        if model_type == "ultraRM":
            return "Human: " + prompt + "\nAssistant: " + response

        if model_type in ["QRLlama", "Gemma", "Qwen", "GPT2Chat"]:
            # If tokenizer has chat template, prefer it
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                toks = tokenizer.apply_chat_template(messages, return_tensors="pt", return_offsets_mapping=True)
                return tokenizer.decode(toks[0])
            # fallback to simple format
            return prompt + "\n" + response

        if model_type == "eurus":
            return "[INST] " + prompt + " [/INST] " + response

        return None

    @staticmethod
    def find_last_consecutive_pair(nums, tokenizer, model_type):
        """
        Find the last index of a special boundary sequence that separates prompt/response.
        This is extremely format-dependent; keep your current rules + allow extension.
        """

        def find_sequence_last_index(lst, sequence):
            n = len(sequence)
            for i in range(len(lst) - n + 1):
                if lst[i : i + n] == sequence:
                    return i + n - 1
            return -1

        def find_last_consecutive_pair_token(ids, tokenizer, text_tokens=None, tokens_id=None):
            if tokens_id is None:
                assert text_tokens is not None
                ids_chat_end = tokenizer(text_tokens, add_special_tokens=False)
                tokens_id = ids_chat_end["input_ids"]
            return find_sequence_last_index(ids.tolist(), tokens_id)

        if model_type == "ultraRM":
            # keep your hard-coded boundary for ultraRM
            return find_last_consecutive_pair_token(
                nums,
                tokenizer,
                tokens_id=[7900, 22137, 29901],  # "\nAssistant:" for that tokenizer
            )

        if model_type == "QRLlama":
            return find_last_consecutive_pair_token(
                nums, tokenizer, text_tokens="assistant<|end_header_id|>\n\n"
            )

        if model_type == "eurus":
            return find_last_consecutive_pair_token(nums, tokenizer, text_tokens="[/INST]")

        # generic fallback: if using chat template, boundary is hard;
        # return -1 means "don't slice" (your caller should handle)
        return -1

    def extract_attention_reward(self, texts_trials: dict, texts_promps: pd.DataFrame):
        attention_trials = {}

        for trial, list_text in texts_trials.items():
            print("trial", trial)

            list_word_original = [str(x) for x in list_text]
            resp_text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]

            prompt = texts_promps[texts_promps["n_resp"] == float(trial)]["prompt_text"].values[0]
            text_chat = self.tokenize_text_chat(self.tokenizer, prompt, resp_text, model_type=self.model_type)

            if text_chat is None:
                # fallback: concatenate prompt/response plainly
                text_chat = prompt + "\n" + resp_text

            input_ids = self.tokenize_text(self.tokenizer, text_chat)

            index_init = self.find_last_consecutive_pair(
                input_ids["input_ids"][0],
                tokenizer=self.tokenizer,
                model_type=self.model_type,
            )

            attention = self.get_attention_model(self.model, input_ids)

            attention_trials[trial] = self.process_attention_reward(
                attention,
                input_ids,
                text=text_chat,
                list_word_original=list_word_original,
                index_init=index_init,
            )

        return attention_trials

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def compute_special_token_idx(tokens_list, special_tokens_ids):
        s = set(special_tokens_ids)
        return [i for i, token in enumerate(tokens_list) if token in s]

    @staticmethod
    def tokenize_text(tokenizer, text: str):
        return tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )

    @staticmethod
    def normalize_rows(matrix):
        cols_normalized = []
        for row in range(matrix.shape[0]):
            elements = matrix[row,]
            cols_normalized.append(elements / (1.0 / (row + 1)))
        return np.array(cols_normalized)

    @staticmethod
    def compute_mean_diagonalbewlow(matrix):
        means = []
        for col in range(matrix.shape[1]):
            valid_elements = matrix[col:, col]
            means.append(np.mean(valid_elements))
        return np.array(means)

    @staticmethod
    def save_attention_np(attention_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = os.path.join(path_folder, f"trial_{trial}")
            os.makedirs(path_folder_trial, exist_ok=True)
            for layer, attention in attention_layer.items():
                np.save(os.path.join(path_folder_trial, f"layer_{layer}.npy"), attention)

    @staticmethod
    def save_attention_df(attention_trials, texts_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = os.path.join(path_folder, f"trial_{trial}")
            os.makedirs(path_folder_trial, exist_ok=True)
            trial_text = texts_trials[trial]
            for layer, attention in attention_layer.items():
                pd.DataFrame({"text": trial_text, "attention": attention}).to_csv(
                    os.path.join(path_folder_trial, f"layer_{layer}.csv"),
                    sep=";",
                    index=False,
                )

    @staticmethod
    def load_attention_np(path_folder):
        attention_trials = {}
        for trial in os.listdir(path_folder):
            trial_path = os.path.join(path_folder, trial)
            if not os.path.isdir(trial_path):
                continue

            attention_layer = {}
            for layer_file in os.listdir(trial_path):
                if not layer_file.endswith(".npy"):
                    continue
                attention = np.load(os.path.join(trial_path, layer_file))
                layer_idx = int(layer_file.split("_")[1].split(".")[0])
                attention_layer[layer_idx] = attention

            attention_trials[float(trial.split("_")[1])] = attention_layer

        return attention_trials

    @staticmethod
    def load_attention_df(path_folder):
        attention_trials = {}
        if not os.path.exists(path_folder):
            print(f"[Missing] {path_folder}")
            return attention_trials

        for trial in os.listdir(path_folder):
            trial_path = os.path.join(path_folder, trial)
            if not os.path.isdir(trial_path):
                continue

            attention_layer = {}
            for layer_file in os.listdir(trial_path):
                if not layer_file.endswith(".csv"):
                    continue
                layer_path = os.path.join(trial_path, layer_file)
                try:
                    df = pd.read_csv(layer_path, sep=";")
                    layer_idx = int(layer_file.split("_")[1].split(".")[0])
                    attention_layer[layer_idx] = df
                except Exception as e:
                    print(f"[skip layer] {layer_path}: {e}")

            if len(attention_layer) == 0:
                print(f"[empty trial] {trial_path}")
                continue

            try:
                trial_idx = float(trial.split("_")[1])
                attention_trials[trial_idx] = attention_layer
            except Exception:
                print(f"[skip trial name] {trial}")
                continue

        if len(attention_trials) == 0:
            print(f"[No trials loaded in] {path_folder}")
        return attention_trials