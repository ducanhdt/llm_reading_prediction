# Reading Time Prediction with LLMs

A PyTorch-based project for predicting reading times using fine-tuned language models with LoRA (Low-Rank Adaptation). This implementation supports Google's Gemma and Qwen models with a custom regression head to predict word-level reading times from eye-tracking data.

**Course Project**: This work is part of the Computational Cognitive Science 3 course at the University of Copenhagen.

## Overview

This project implements a reading time prediction model based on the methodology from the "Reverse-Engineering the Reader" paper. The model:
- Fine-tunes pre-trained language models (Gemma or Qwen) using LoRA for efficient adaptation
- Predicts reading times at the word level using either:
  - Direct regression from hidden states
  - Surprisal-based features (surprisal, word length, word frequency)
- Incorporates KL divergence regularization to maintain language modeling capabilities
- Trains on eye-tracking datasets from OASST-ETC (Open Assistant Eye-Tracking Corpus)

### Supported Models
- **Google Gemma** (gemma-3-270m)
- **Qwen** models

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using PEFT
- **Dual Prediction Modes**: 
  - Direct regression head
  - Surprisal-based feature prediction
- **KL Divergence Regularization**: Prevents model drift from pre-trained behavior
- **Word-level Alignment**: Automatic tokenization and alignment of reading times with subword tokens
- **Configurable Training**: JSON-based configuration for easy experimentation

## Project Structure

```
├── model.py # GemmaWithRegressionHead model definition
├── trainer.py # Training script and configuration loader
├── data_reader.py # Data loading and preprocessing utilities
├── sample_config.json # Example training configuration
├── data/ # Dataset directory
│ ├── oasstetc_all_train.csv
│ ├── oasstetc_all_test.csv
│ ├── oasstetc_complete_train.csv
│ └── oasstetc_complete_test.csv
│
├── Evaluation/ # Evaluation protocols and analysis
│ ├── explicit_eval/ # Explicit preference-based evaluation
│ │ ├── build_triples.py # Build (prompt, chosen, rejected) triples
│ │ ├── model_registry.py # Unified registry for baseline / finetuned / PEFT models
│ │ ├── explicit_eval.py # Preference judgment and accuracy computation
│ │ ├── main_explicit_eval.py # Entry point for explicit evaluation
│ │ └── plot_accuracy.py # Accuracy visualization
│ │
│ └── implicit_eval/ # Implicit cognitive alignment evaluation
│ ├── model_loader.py # Model loading utilities
│ ├── model_att.py # Model attention extraction
│ ├── compare_att.py # Correlation & statistical comparison
│ ├── plotter.py # Plotting utilities
│ ├── main_compute_attention.py # Compute model attention signals
│ ├── main_compare_trials.py # Trial-level correlation analysis
│ ├── main_plot_attention_layers.py
│ └── main_plot_chosen_rejected.py
```

## Installation

### Requirements

```bash
pip install torch transformers peft datasets pandas wordfreq
```

### Optional (for experiment tracking)
```bash
pip install wandb
```

## Usage

### 1. Prepare Your Data

The data should be in CSV format with the following columns:
- `word`: The word text
- `sentence_num`: Sentence identifier for grouping words
- Target column (e.g., `first_fix_dur`, `reading_times`): Reading time values

### 2. Configure Training

Edit `sample_config.json` or create your own configuration file:

```json
{
    "base_model": "google/gemma-3-270m",
    "train_data_path": "data/oasstetc_all_train.csv",
    "eval_data_path": "data/oasstetc_all_test.csv",
    "KL_alpha": 500,
    "use_surprisal_loss": true,
    "use_wandb": false,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "learning_rate": 2e-5,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 2,
    "total_steps": 1000
}
```

### 3. Train the Model

```python
from trainer import train_model

# Train with your config
train_model("sample_config.json")
```

Or run directly:
```bash
python trainer.py
```

## Model Architecture

### GemmaWithRegressionHead

The model consists of:

1. **Base Model**: Pre-trained Gemma language model with LoRA adapters
   - LoRA rank: 8
   - LoRA alpha: 32
   - Target modules: q_proj, v_proj

2. **Prediction Heads**:
   - **Regression Head**: Linear layer mapping hidden states to reading times
   - **Surprisal Head**: Linear layer combining 3 features (surprisal, word length, log frequency)

3. **Loss Components**:
   - MSE Loss: Reading time prediction error
   - KL Divergence: Regularization term to maintain language modeling capabilities

### Key Methods

- `compute_surprisal()`: Computes token-level surprisal values
- `predict_with_features()`: Predicts reading times from linguistic features
- `compute_kl_metrics()`: Calculates KL divergence between adapted and base model

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_model` | HuggingFace model identifier | `google/gemma-3-270m` |
| `KL_alpha` | Weight for KL divergence loss | `500` |
| `use_surprisal_loss` | Use surprisal-based prediction | `true` |
| `use_wandb` | Enable Weights & Biases logging | `false` |
| `num_train_epochs` | Number of training epochs | `3` |
| `per_device_train_batch_size` | Batch size per device | `1` |
| `learning_rate` | Learning rate | `2e-5` |
| `max_grad_norm` | Gradient clipping threshold | `1.0` |
| `warmup_steps` | Number of warmup steps | `50` |
| `total_steps` | Maximum training steps | `1000` |
| `target_feature` | Column name for reading times | `first_fix_dur` |

## Data Format

Input CSV files should have this structure:

```csv
word,sentence_num,first_fix_dur
The,1,250.0
cat,1,180.0
sat,1,200.0
on,1,150.0
the,1,120.0
mat,1,220.0
```

The model will:
1. Group words by `sentence_num`
2. Tokenize each sentence
3. Align reading times with subword tokens
4. Calculate word-level features (length, frequency)

## Training Process

1. **Data Loading**: CSV files are read and grouped into sentences
2. **Tokenization**: Words are tokenized and aligned with reading time labels
3. **Feature Extraction**: Word length and frequency features are computed
4. **Model Training**: LoRA-adapted model is trained with combined loss:
   - MSE loss for reading time prediction
   - KL divergence for regularization
5. **Evaluation**: Model is evaluated on held-out test set

## Outputs

The trained model is saved to `./gemma-reading-time/` with:
- Model checkpoints (every `save_steps`)
- Training logs
- Best model based on evaluation loss

## Evaluation

This project adopts a two-level evaluation protocol that distinguishes
**implicit cognitive alignment** from **explicit preference-based performance**.
The goal is to analyze whether training signals derived from eye-tracking data
affect both latent reading behavior and overt decision-making.

All evaluation code is organized under the `Evaluation/` directory.


## Citation

If you use this code in your research, please consider citing:

**Reverse-Engineering the Reader:**
```
@misc{kiegeland2024reverseengineeringreader,
  title={Reverse-Engineering the Reader}, 
  author={Samuel Kiegeland and Ethan Gotlieb Wilcox and Afra Amini and David Robert Reich and Ryan Cotterell},
  year={2024},
  eprint={2410.13086},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.13086}
}
```

**OASST-ETC (Open Assistant Eye-Tracking Corpus):**
```
@article{deng2024oasst,
  title={OASST-ETC: A Large-Scale Eye-Tracking Corpus for Human-AI Conversation},
  author={Deng, Jiajie and Hollenstein, Nora},
  journal={arXiv preprint},
  year={2024}
}
```

**Additional relevant papers:**
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Gemma: Google's open language models
- Surprisal-based reading time prediction

## License

This project is for academic research purposes.

## Acknowledgments

- Based on methodology from "Reverse-Engineering the Reader" (Wilcox et al., 2023)
- Dataset from OASST-ETC: Open Assistant Eye-Tracking Corpus (Deng & Hollenstein, 2024)
- Built with HuggingFace Transformers and PEFT
- Uses wordfreq for frequency estimation
- Part of Computational Cognitive Science 3 course at University of Copenhagen

## Contact

For questions or issues, please open an issue on the GitHub repository.
