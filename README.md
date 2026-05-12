# Fine-tuning Cohere Transcribe with Hugging Face Datasets

This repository is a minimal example of how I fine-tuned `CohereLabs/cohere-transcribe-03-2026` with datasets from the Hugging Face Hub.

## Files

```text
train/fine-tune_on_hf_dataset_all_config.py   # fine-tuning script
evaluate/evaluation_on_hf_dataset_cohere.py   # evaluation script
pyproject.toml                                # dependencies
uv.lock                                       # locked dependency versions
```

Local artifacts such as `outputs/`, `wandb/`, `predictions_dir/`, and exported model folders are not meant to be committed.

## Setup

```bash
uv sync
uvx hf auth login
```

## Train

```bash
uv run torchrun --nproc_per_node=4 train/fine-tune_on_hf_dataset_all_config.py \
  --model_name CohereLabs/cohere-transcribe-03-2026 \
  --language en \
  --train_datasets your-org/your-train-dataset \
  --eval_datasets your-org/your-eval-dataset \
  --sampling_rate 16000 \
  --num_proc 8 \
  --train_strategy epoch \
  --learning_rate 2e-4 \
  --train_batchsize 4 \
  --eval_batchsize 16 \
  --num_epochs 5 \
  --output_dir outputs/lr_2e-4 \
  --gradient_accumulation_steps 32
```

The script loads every config from each dataset passed to `--train_datasets`, concatenates them, and evaluates on the `test` split of every config from `--eval_datasets`.

## Things to Change for Your Dataset

- `--language`: set the Cohere Transcribe language prompt for your data.
- `--train_datasets` / `--eval_datasets`: replace these with your Hugging Face dataset IDs.
- Dataset splits: the script currently expects `train` for training and `test` for evaluation.
- Audio column: the script expects an `audio` column.
- Transcript column: the script currently renames `hanzi` to `sentence`; change this to your transcript column name.
- Text normalization: adjust punctuation, casing, spacing, and other cleanup rules for your benchmark.
- Space handling: the script currently removes spaces from transcripts; do not do this if your language needs word boundaries.
- Length filtering: adjust min/max audio duration and max label length if needed.
- Metric: the code uses CER; add WER if that is more appropriate for your language.
- Batch sizes / gradient accumulation: tune these for your GPU memory.

## Evaluate

Evaluate a base model or fine-tuned checkpoint with:

```bash
uv run python evaluate/evaluation_on_hf_dataset_cohere.py \
  --hf_model outputs/lr_2e-4/checkpoint-2877 \
  --dataset your-org/your-dataset \
  --configs your_config \
  --split test \
  --language en \
  --batch_size 8 \
  --output_dir predictions_dir \
  --log_normalized true
```

Useful options:

- `--hf_model`: Hugging Face model ID or local checkpoint path.
- `--configs`: dataset configs to evaluate.
- `--split`: dataset split, default `test`.
- `--max_samples`: run a quick smoke test.
- `--max_duration`: skip long audio clips.
- `--max_new_tokens`: control generation length.
- `--log_normalized`: log normalized alignment instead of raw alignment.

The output file includes raw CER, normalized CER, and per-sample alignment logs.
