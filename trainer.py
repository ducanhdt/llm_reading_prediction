from transformers import TrainingArguments, Trainer
from model import GemmaWithRegressionHead
from data_reader import read_data, tokenize_and_align_labels, DataCollatorForReadingTime
import os
from datasets import Dataset
import json


def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions
	logits = preds[0]
	mse_loss = preds[1]
	kl_loss = preds[2]
	avg_log_prob = preds[3]
	delta_log_prob = preds[4]
	return {
		"mse_loss": mse_loss.mean(),
		"kl_loss": kl_loss.mean(),
		"avg_log_prob": avg_log_prob.mean(),
		"delta_log_prob": delta_log_prob.mean(),
	}


def train_model(config_path):
	# Load parameters from config file
	with open(config_path, "r") as f:
		config = json.load(f)

	# Model parameters
	base_model = config.get("base_model", "google/gemma-3-270m")
	KL_alpha = config.get("KL_alpha", 500)
	use_surprisal_loss = config.get("use_surprisal_loss", True)
	use_wandb = config.get("use_wandb", True)

	# Training parameters
	num_train_epochs = config.get("num_train_epochs", 3)
	per_device_train_batch_size = config.get("per_device_train_batch_size", 4)
	learning_rate = config.get("learning_rate", 2e-5)
	max_grad_norm = config.get("max_grad_norm", 1.0)
	warmup_steps = config.get("warmup_steps", 50)
	logging_steps = config.get("logging_steps", 10)
	eval_steps = config.get("eval_steps", 100)
	save_steps = config.get("save_steps", 100)
	save_total_limit = config.get("save_total_limit", 2)
	total_steps = config.get("total_steps", 1000)
	train_path = config.get("train_path", "data/oasstetc_all_train.csv")
	eval_path = config.get("eval_path", "data/oasstetc_all_test.csv")
	target_feature = config.get("target_feature", "first_fix_dur")
	push_to_hub = config.get("push_to_hub", True)

	model = GemmaWithRegressionHead(
		base_model, KL_alpha=KL_alpha, use_surprisal_loss=use_surprisal_loss
	)
	df_eval = read_data(eval_path, target_feature)
	df_train = read_data(train_path, target_feature)
	dataset = Dataset.from_pandas(df_train)
	dataset_test = Dataset.from_pandas(df_eval)

	# Tokenize the dataset
	tokenized_dataset = dataset.map(
		tokenize_and_align_labels,
		batched=True,
		fn_kwargs={"tokenizer": model.tokenizer},
	)
	tokenized_dataset_test = dataset_test.map(
		tokenize_and_align_labels,
		batched=True,
		fn_kwargs={"tokenizer": model.tokenizer},
	)
	# Print dataset info to debug
	print(f"Dataset size: {len(tokenized_dataset)}")
	print(f"Dataset test size: {len(tokenized_dataset_test)}")
	print(f"Sample tokenized input: {tokenized_dataset[0]}")
	# Set include_word_features=True if using use_surprisal_loss=True
	data_collator = DataCollatorForReadingTime(
		tokenizer=model.tokenizer, include_word_features=True
	)
	if "gemma" in base_model.lower():
		output_name = f"gemma3-1b-reading-time-KL-{KL_alpha}-lr-{learning_rate}"
	else:
		output_name = f"qwen0.6b-reading-time-KL-{KL_alpha}-lr-{learning_rate}"
  
	training_args = TrainingArguments(
		output_dir=output_name,
		num_train_epochs=num_train_epochs,
		per_device_train_batch_size=per_device_train_batch_size,
		learning_rate=learning_rate,
		max_grad_norm=max_grad_norm,  # Add gradient clipping to prevent instability
		warmup_steps=warmup_steps,  # Warmup for training stability
		logging_steps=logging_steps,
		eval_strategy="steps",
		eval_steps=eval_steps,
		save_steps=save_steps,
		save_total_limit=save_total_limit,
		save_strategy="steps",
		save_safetensors=False,
		report_to="none" if not use_wandb else ["wandb"],
		logging_first_step=True,
		eval_on_start=True,
		max_steps=total_steps,
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset,
		eval_dataset=tokenized_dataset_test,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		# callbacks=[MetricsCallback()]
	)

	if os.environ["WANDB_DISABLED"] == "false":
		import wandb

		wandb.login(key="5dcf43d42b9e34786a216356f212ffb5fd2656c1")

		# Initialize wandb
		wandb.init(
			project="gemma-reading-time",
			config=config,
			name=output_name,
		)

	print("start Training")
	trainer.train()

	if push_to_hub:
		model.base_model.push_to_hub(
			output_name,
			use_safetensors=True,
		)

	return model


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Train Gemma model for reading time prediction."
	)
	parser.add_argument(
		"--config",
		type=str,
		help="Path to the config file.",
		default="sample_config.json",
	)

	args = parser.parse_args()
	print(f"Using config file: {args.config}")
	train_model(args.config)
