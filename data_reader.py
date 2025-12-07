# Load model directly

from transformers import AutoTokenizer

from datasets import Dataset
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd
import math
from wordfreq import word_frequency

def read_data(file_path, target_column='reading_times'):
	data = pd.read_csv(file_path) # each word on a row, num_sentence contains sentence_index
	# remove rows with NaN in target_column and 'word' column
	data = data.dropna(subset=[target_column, 'word'])
	sentences = data.groupby('sentence_num')['word'].apply(list).tolist()
	reading_times = data.groupby('sentence_num')[target_column].apply(list).tolist()
	for i in range(len(sentences)):
		assert len(sentences[i]) == len(reading_times[i]), f"Length mismatch in sentence {i}"
	return pd.DataFrame({'text': sentences, 'reading_times': reading_times})

def tokenize_and_align_labels(examples, tokenizer=None):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)

    labels = []
    word_lengths_batch = []
    log_unigram_frequencies_batch = []
    
    for i, label in enumerate(examples[f"reading_times"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        words = examples["text"][i]
        
        # Calculate word lengths and log unigram frequencies for this example
        lengths = [len(word) for word in words]
        freqs = [word_frequency(word, "en", wordlist="best") for word in words]
        log_freqs = [
            -math.log2(frequency if frequency > 0 else 1e-10) for frequency in freqs
        ]
        
        previous_word_idx = None
        label_ids = []
        length_ids = []
        log_freq_ids = []
        
        for word_idx in word_ids:  # Set the special tokens to -100 or 0.0
            if word_idx is None:
                label_ids.append(0.0)
                length_ids.append(0)
                log_freq_ids.append(0.0)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
                length_ids.append(lengths[word_idx])
                log_freq_ids.append(log_freqs[word_idx])
            else:
                label_ids.append(0.0)
                length_ids.append(0)
                log_freq_ids.append(0.0)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
        word_lengths_batch.append(length_ids)
        log_unigram_frequencies_batch.append(log_freq_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_lengths"] = word_lengths_batch
    tokenized_inputs["log_unigram_frequencies"] = log_unigram_frequencies_batch
    return tokenized_inputs

@dataclass
class DataCollatorForReadingTime:
	tokenizer: Any
	padding: bool = True
	max_length: int = None
	pad_to_multiple_of: int = None
	include_word_features: bool = False  # Set to True to include word_ids, word_lengths, log_unigram_frequencies
	
	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		# Separate special fields from input features
		labels = [feature.pop("labels") for feature in features]
		
		# Extract word features if they exist
		word_lengths_list = None
		log_freqs_list = None
		if self.include_word_features and "word_lengths" in features[0]:
			word_lengths_list = [feature.pop("word_lengths") for feature in features]
			log_freqs_list = [feature.pop("log_unigram_frequencies") for feature in features]
		
		# Pad input features (input_ids, attention_mask)
		# Set pad token if not already set
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		batch = self.tokenizer.pad(
			features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt"
		)
		
		# Pad labels to match the padded input length
		max_label_length = max(len(label) for label in labels)
		padded_labels = []
		for label in labels:
			padding_length = max_label_length - len(label)
			padded_label = label + [-1.0] * padding_length  # Use -1 for padding
			padded_labels.append(padded_label)
		
		batch["labels"] = torch.tensor(padded_labels, dtype=torch.float)
		
		# Add word features if they exist
		if self.include_word_features and word_lengths_list is not None:
			# Create word_ids from labels - find first token of each word
			word_ids_batch = []
			word_lengths_batch = []
			log_freqs_batch = []
			
			for i, (feature_labels, word_lengths, log_freqs) in enumerate(zip(labels, word_lengths_list, log_freqs_list)):
				# Build word_ids: map each token to its word index
				word_ids = []
				word_idx = -1
				
				for j, label_val in enumerate(feature_labels):
					if label_val != 0.0:  # First token of a new word
						word_idx += 1
						word_ids.append(word_idx)
					else:
						word_ids.append(-1)  # Padding or continuation token
				
				# Pad word_ids to match max_label_length
				word_ids.extend([-1] * (max_label_length - len(word_ids)))
				word_ids_batch.append(word_ids)
				
				# Extract actual word features (non-zero values)
				actual_lengths = []
				actual_freqs = []
				for wl, lf in zip(word_lengths, log_freqs):
					if wl != 0:
						actual_lengths.append(wl)
						actual_freqs.append(lf)
				
				word_lengths_batch.append(actual_lengths)
				log_freqs_batch.append(actual_freqs)
			
			# Pad word features to max word count
			max_words = max(len(wl) for wl in word_lengths_batch)
			padded_word_lengths = []
			padded_log_freqs = []
			
			for lengths, freqs in zip(word_lengths_batch, log_freqs_batch):
				padding_length = max_words - len(lengths)
				padded_word_lengths.append(lengths + [-1] * padding_length)
				padded_log_freqs.append(freqs + [-1.0] * padding_length)
			
			batch["word_ids"] = torch.tensor(word_ids_batch, dtype=torch.long)
			batch["word_lengths"] = torch.tensor(padded_word_lengths, dtype=torch.long)
			batch["log_unigram_frequencies"] = torch.tensor(padded_log_freqs, dtype=torch.float)
		
		return batch
if __name__ == "__main__":
    print("Testing data reader...")
    df_test = read_data('data/datasets/zuco_test.csv', 'first_fix_dur')
    # df_test = read_data('data/datasets/zuco_test.csv', 'first_fix_dur').sample(100)
    # df_train = read_data('data/datasets/zuco_train.csv', 'first_fix_dur')
    df_train = read_data('data/data_splits/original_data/oasstetc_complete.csv', 'first_fix_dur')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

    dataset = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, tokenizer=tokenizer)
    tokenized_dataset_test = dataset_test.map(tokenize_and_align_labels, batched=True, tokenizer=tokenizer)
    # Print dataset info to debug
    print(f"Dataset size: {len(tokenized_dataset)}")
    print(f"Dataset test size: {len(tokenized_dataset_test)}")
    print(f"Sample tokenized input: {tokenized_dataset[0]}")
    # Set include_word_features=True if using use_surprisal_loss=True
    data_collator = DataCollatorForReadingTime(tokenizer=tokenizer, include_word_features=True)


