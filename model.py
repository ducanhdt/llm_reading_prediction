import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM 

from peft import LoraConfig, get_peft_model, TaskType


class GemmaWithRegressionHead(nn.Module):
	def __init__(self, base_model, KL_alpha=0.3, mse_weight=1.0, use_surprisal_loss=False):
		super().__init__()
		self.KL_alpha = KL_alpha
		self.mse_weight = mse_weight
		self.use_surprisal_loss = use_surprisal_loss
		
		self.tokenizer = AutoTokenizer.from_pretrained(base_model)
		self.base_model = AutoModelForCausalLM.from_pretrained(base_model)
		
		self.lora_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			r=8,
			lora_alpha=32,
			lora_dropout=0.1,
			target_modules=["q_proj", "v_proj"]
		)

		self.base_model = get_peft_model(self.base_model, self.lora_config)

  
		if not use_surprisal_loss:
			self.regression_head = nn.Linear(self.base_model.config.hidden_size, 1)
		else:
			# Linear layer for surprisal-based prediction: 3 features (surprisal, length, frequency) + bias
			self.surprisal_head = nn.Linear(3, 1)
	
	def predict_with_features(self, surprisal, word_lengths, word_frequencies):
		"""Use nn.Linear to predict reading times from features"""
		# Stack features: [batch, 3] where 3 = [surprisal, word_length, word_frequency]
		features = torch.stack([surprisal, word_lengths, word_frequencies], dim=1)
		predictions = self.surprisal_head(features).squeeze(-1)
		return predictions
	
	def compute_surprisal(self, input_ids, attention_mask, word_ids):
		"""Compute surprisal values"""
		# Add BOS token manually - match batch size
		batch_size = input_ids.size(0)
		bos_tensor = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.base_model.device)
		input_ids_with_bos = torch.cat([bos_tensor, input_ids], dim=1)
		
		attention_mask_with_bos = torch.cat([
			torch.ones((batch_size, 1), dtype=torch.int64, device=self.base_model.device),
			attention_mask
		], dim=1)
		
		# Create labels mask
		mask = word_ids != -1
		mask_with_bos = torch.cat([
			torch.ones((batch_size, 1), dtype=torch.bool, device=self.base_model.device),
			mask
		], dim=1)
		labels = torch.where(mask_with_bos, input_ids_with_bos, -100)
		
		# Get model outputs
		outputs = self.base_model(
			input_ids=input_ids_with_bos,
			attention_mask=attention_mask_with_bos,
			labels=labels
		)
		
		# Compute surprisal
		log_probs = torch.log_softmax(outputs.logits, dim=-1)
		log_probs_shifted = log_probs[:, :-1, :]
		input_ids_shifted = input_ids_with_bos[:, 1:]
		
		surprisal = torch.mul(
			-1,
			torch.gather(log_probs_shifted, 2, input_ids_shifted[:, :, None]).squeeze(-1)
		)
		
		return surprisal, outputs.loss, log_probs
	
	def compute_kl_metrics(self, input_ids, attention_mask, word_ids, log_probs):
		"""Compute KL divergence and log probability metrics"""
		with torch.no_grad(), self.base_model.disable_adapter():
			_, _, ref_log_probs = self.compute_surprisal(input_ids, attention_mask, word_ids)
		
		# Compute full KL divergence
		full_kl_mask = word_ids != -1
		# Create a batch-sized False tensor for the first position
		batch_size = full_kl_mask.size(0)
		first_token_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.base_model.device)
		full_kl_mask = torch.cat([first_token_mask, full_kl_mask[:, :-1]], dim=1)
		
		log_probs_sliced = log_probs[:, :-1, :]
		ref_log_probs_sliced = ref_log_probs[:, :-1, :]
		
		# KL divergence: KL(target || ref)
		kl_divergence = torch.nn.functional.kl_div(
			log_probs_sliced[full_kl_mask].unsqueeze(0),
			ref_log_probs_sliced[full_kl_mask].unsqueeze(0),
			reduction="none",
			log_target=True
		).sum(-1)
		kl_loss = kl_divergence.mean()
		
		# Calculate average log probability metrics
		shift_input_ids = input_ids[..., 1:].contiguous()
		shift_attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else torch.ones_like(input_ids[..., 1:])
		answer_mask = shift_attention_mask.float()
		
		# Shift log probs for next token prediction
		log_probs_for_tokens = log_probs[:, :-1, :]
		ref_log_probs_for_tokens = ref_log_probs[:, :-1, :]
		
		# Get log probs for actual next tokens
		gathered_log_probs = torch.gather(
			log_probs_for_tokens,
			dim=-1,
			index=shift_input_ids.unsqueeze(-1)
		).squeeze(-1)
		
		gathered_ref_log_probs = torch.gather(
			ref_log_probs_for_tokens,
			dim=-1,
			index=shift_input_ids.unsqueeze(-1)
		).squeeze(-1)
		
		# Calculate average log probability over non-padding tokens
		avg_log_prob = (gathered_log_probs * answer_mask).sum() / answer_mask.sum()
		delta_log_prob = avg_log_prob - (gathered_ref_log_probs * answer_mask).sum() / answer_mask.sum()
		
		return kl_loss, avg_log_prob, delta_log_prob
	
	def forward(self, input_ids, attention_mask=None, labels=None, word_ids=None, 
				word_lengths=None, log_unigram_frequencies=None, return_dict=True, **kwargs):
		# Check if we're in training mode with regression labels
		is_regression_task = labels is not None and isinstance(labels, torch.Tensor) and labels.dtype == torch.float
		
		if is_regression_task and self.use_surprisal_loss:
			# Use surprisal-based loss with nn.Linear
			if word_ids is None or word_lengths is None or log_unigram_frequencies is None:
				raise ValueError("word_ids, word_lengths, and log_unigram_frequencies required for surprisal loss")
			
			# Compute surprisal
			surprisal, ce_loss, log_probs = self.compute_surprisal(input_ids, attention_mask, word_ids)
			
			# Mask and sum surprisal by word
			mask_word_ids = word_ids != -1
			masked_surprisal = surprisal * mask_word_ids
			masked_word_ids = word_ids * mask_word_ids
			
			summed_surprisal = torch.zeros(
				surprisal.shape[0], surprisal.shape[1], device=surprisal.device
			)
			summed_surprisal.scatter_add_(1, masked_word_ids, masked_surprisal)
			
			# Prepare masks for valid words - labels are token-level
			mask_words = labels != -1
			masked_reading_times = labels[mask_words == 1]
			
			# Map word-level features to token positions using word_ids
			# Expand word features to token positions
			batch_size, seq_len = word_ids.shape
			_, max_words = word_lengths.shape
			
			# Create expanded tensors for word features at token positions
			expanded_lengths = torch.zeros(batch_size, seq_len, device=word_lengths.device, dtype=torch.float)
			expanded_frequencies = torch.zeros(batch_size, seq_len, device=log_unigram_frequencies.device, dtype=torch.float)
			
			# Map word features to their corresponding token positions
			for b in range(batch_size):
				for t in range(seq_len):
					word_idx = word_ids[b, t].item()
					if word_idx >= 0 and word_idx < max_words:
						if word_lengths[b, word_idx] != -1:  # Valid word feature
							expanded_lengths[b, t] = word_lengths[b, word_idx].float()
							expanded_frequencies[b, t] = log_unigram_frequencies[b, word_idx]
			
			# Now extract features at valid positions
			masked_lengths = expanded_lengths[mask_words == 1]
			masked_frequencies = expanded_frequencies[mask_words == 1]
			masked_surprisal_words = summed_surprisal[mask_words == 1]
			
			# Use nn.Linear for prediction
			predictions_trg = self.predict_with_features(
				masked_surprisal_words,
				masked_lengths,
				masked_frequencies
			)
			
			# Compute MSE loss
			mse_loss = torch.nn.functional.mse_loss(masked_reading_times, predictions_trg)
			loss = self.mse_weight * mse_loss
			
			# Compute KL divergence and log probability metrics if needed
			if self.KL_alpha > 0:
				kl_loss, avg_log_prob, delta_log_prob = self.compute_kl_metrics(
					input_ids, attention_mask, word_ids, log_probs
				)
				loss = loss + self.KL_alpha * kl_loss
			else:
				kl_loss = torch.tensor([0.0], device=loss.device)
				avg_log_prob = torch.tensor([0.0], device=loss.device)
				delta_log_prob = torch.tensor([0.0], device=loss.device)
			
			# Ensure all metrics have shape [1] to avoid gather warning
			return {
				"loss": loss,
				"logits": predictions_trg,
				"mse_loss": mse_loss.unsqueeze(0) if mse_loss.dim() == 0 else mse_loss,
				"kl_loss": kl_loss.unsqueeze(0) if kl_loss.dim() == 0 else kl_loss,
				"avg_log_prob": avg_log_prob.unsqueeze(0) if avg_log_prob.dim() == 0 else avg_log_prob,
				"delta_log_prob": delta_log_prob.unsqueeze(0) if delta_log_prob.dim() == 0 else delta_log_prob,
			}
		
		elif is_regression_task:
			# Original regression head approach
			outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
			hidden_states = outputs.hidden_states[-1]
			predictions = self.regression_head(hidden_states).squeeze(-1)
			
			loss_fct = nn.MSELoss()
			active_loss = labels != 0.0
			active_predictions = predictions[active_loss]
			active_labels = labels[active_loss]
			mse_loss = loss_fct(active_predictions, active_labels)
			
			loss = self.mse_weight * mse_loss
			
			if self.KL_alpha > 0:
				with torch.no_grad(), self.base_model.disable_adapter():
					ref_outputs = self.base_model(
						input_ids=input_ids,
						attention_mask=attention_mask,
					)
				ref_logits = ref_outputs.logits
				logits = outputs.logits
				shift_logits = logits[..., :-1, :].contiguous()
				shift_ref_logits = ref_logits[..., :-1, :].contiguous()
				
				shift_attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else torch.ones_like(input_ids[..., 1:])
				answer_mask = shift_attention_mask.float()

				log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
				ref_log_probs = torch.nn.functional.log_softmax(shift_ref_logits, dim=-1)

				kl_divergence = torch.exp(ref_log_probs) * (ref_log_probs - log_probs)

				kl_loss_per_token = (kl_divergence.sum(dim=-1) * answer_mask)
				kl_loss = kl_loss_per_token.sum() / answer_mask.sum()
				loss = loss + self.KL_alpha * kl_loss
	
				shift_input_ids = input_ids[..., 1:].contiguous()
				
				gathered_log_probs = torch.gather(
					log_probs, 
					dim=-1, 
					index=shift_input_ids.unsqueeze(-1)
				).squeeze(-1)
				
				gathered_ref_log_probs = torch.gather(
					ref_log_probs,
					dim=-1,
					index=shift_input_ids.unsqueeze(-1)
				).squeeze(-1)
				
				avg_log_prob = (gathered_log_probs * answer_mask).sum() / answer_mask.sum()
				delta_log_prob = avg_log_prob - (gathered_ref_log_probs * answer_mask).sum() / answer_mask.sum()
			else:
				kl_loss = torch.tensor([0.0], device=loss.device)
				avg_log_prob = torch.tensor([0.0], device=loss.device)
				delta_log_prob = torch.tensor([0.0], device=loss.device)
			
			# Ensure all metrics have shape [1] to avoid gather warning
			return {
				"loss": loss, 
				"logits": predictions, 
				"mse_loss": mse_loss.unsqueeze(0) if mse_loss.dim() == 0 else mse_loss, 
				"kl_loss": kl_loss.unsqueeze(0) if kl_loss.dim() == 0 else kl_loss,
				"avg_log_prob": avg_log_prob.unsqueeze(0) if avg_log_prob.dim() == 0 else avg_log_prob,
				"delta_log_prob": delta_log_prob.unsqueeze(0) if delta_log_prob.dim() == 0 else delta_log_prob,
			}
		else:
			# Inference mode or standard LM loss
			return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=return_dict, **kwargs)
	
	def generate(self, *args, **kwargs):
		# Delegate generation to base model
		return self.base_model.generate(*args, **kwargs)
