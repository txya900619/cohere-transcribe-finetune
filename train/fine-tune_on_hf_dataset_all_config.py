import argparse
import contextlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from torch import nn
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(
    description="Fine-tuning script for Cohere ASR models."
)
parser.add_argument(
    "--model_name",
    type=str,
    required=False,
    default="CohereLabs/cohere-transcribe-03-2026",
    help="Hugging Face model name to fine-tune. Eg: CohereLabs/cohere-transcribe-03-2026",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="zh",
    help="Decoder prompt language code. Eg: en, zh, ja.",
)
parser.add_argument(
    "--sampling_rate",
    type=int,
    required=False,
    default=16000,
    help="Sampling rate of audios.",
)
parser.add_argument(
    "--num_proc",
    type=int,
    required=False,
    default=8,
    help="Number of parallel jobs to run. Helps parallelize the dataset prep stage.",
)
parser.add_argument(
    "--train_strategy",
    type=str,
    required=False,
    default="steps",
    help="Training strategy. Choose between steps and epoch.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    required=False,
    default=2e-4,
    help="Learning rate for the fine-tuning process.",
)

parser.add_argument(
    "--train_batchsize",
    type=int,
    required=False,
    default=4,
    help="Batch size during the training phase.",
)
parser.add_argument(
    "--eval_batchsize",
    type=int,
    required=False,
    default=16,
    help="Batch size during the evaluation phase.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    required=False,
    default=5,
    help="Number of epochs to train for.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    required=False,
    default=100000,
    help="Number of steps to train for.",
)
parser.add_argument(
    "--resume_from_ckpt",
    type=str,
    required=False,
    default=None,
    help="Path to a trained checkpoint to resume training from.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    default="output_model_dir",
    help="Output directory for the checkpoints generated.",
)
parser.add_argument(
    "--train_datasets",
    type=str,
    nargs="+",
    required=True,
    default=[],
    help="List of datasets to be used for training.",
)

parser.add_argument(
    "--eval_datasets",
    type=str,
    nargs="+",
    required=True,
    default=[],
    help="List of datasets to be used for evaluation.",
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    required=False,
    default=32,
    help="Number of gradient accumulation steps.",
)

args = parser.parse_args()

if args.train_strategy not in ["steps", "epoch"]:
    raise ValueError("The train strategy should be either steps and epoch.")

if len(args.train_datasets) == 0:
    raise ValueError("No train dataset has been passed")
if len(args.eval_datasets) == 0:
    raise ValueError("No evaluation dataset has been passed")

print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
print("ARGUMENTS OF INTEREST:")
print(vars(args))
print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

gradient_checkpointing = False
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


#############################       MODEL LOADING       #####################################
processor = AutoProcessor.from_pretrained(args.model_name)
processor.get_decoder_prompt_ids(language=args.language, punctuation=True)

model = CohereAsrForConditionalGeneration.from_pretrained(
    args.model_name,
    # attn_implementation="sdpa",
)


def plain_cross_entropy_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch=None,
    ignore_index: int = -100,
    **kwargs,
):
    logits = logits.float()
    labels = labels.to(logits.device)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


model._loss_function = plain_cross_entropy_loss

if (
    model.config.decoder_start_token_id is None
    and model.generation_config.decoder_start_token_id is not None
):
    model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id

if (
    model.generation_config.decoder_start_token_id is None
    and model.config.decoder_start_token_id is not None
):
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id


if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


if gradient_checkpointing:
    model.config.use_cache = False


def remove_trailing_comma_and_add_dot(x):
    sentence = re.sub(r"，$", "", x["sentence"])
    if sentence[-1] not in ["。", "？", "！"]:
        sentence += "。"
    return {"sentence": sentence}


############################        DATASET LOADING AND PREP        ##########################


def load_all_datasets(split):
    combined_dataset = []
    if split == "train":
        for i, ds in enumerate(args.train_datasets):
            for config_name in get_dataset_config_names(ds):
                dataset = load_dataset(
                    ds,
                    config_name,
                    split="train",
                )

                none_audio_dataset = dataset.filter(lambda x: x["audio"] is None)
                if len(none_audio_dataset) > 0:
                    print(
                        f"Dataset {ds} with config {config_name} has {len(none_audio_dataset)} samples with no audio."
                    )
                    for sample in none_audio_dataset:
                        print(f"id {sample['id']} no audio")
                    dataset = dataset.filter(lambda x: x["audio"] is not None)

                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                dataset = dataset.rename_column("hanzi", "sentence")
                dataset = dataset.remove_columns(
                    set(dataset.features.keys()) - set(["audio", "sentence"])
                )
                dataset = dataset.map(remove_trailing_comma_and_add_dot, num_proc=8)
                combined_dataset.append(dataset)
    elif split == "eval":
        for i, ds in enumerate(args.eval_datasets):
            for config_name in get_dataset_config_names(ds):
                if "test" not in get_dataset_split_names(ds, config_name):
                    continue
                dataset = load_dataset(ds, config_name, split="test")
                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                dataset = dataset.rename_column("hanzi", "sentence")
                dataset = dataset.remove_columns(
                    set(dataset.features.keys()) - set(["audio", "sentence"])
                )

                dataset = dataset.map(remove_trailing_comma_and_add_dot, num_proc=8)
                combined_dataset.append(dataset)

    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return


max_label_length = model.config.max_position_embeddings
min_input_length = 3.0
max_input_length = 35.0


def is_in_length_range(audio, sentence):
    input_length = len(audio["array"]) / audio["sampling_rate"]
    label_length = len(processor.tokenizer.tokenize(sentence))
    return (
        min_input_length < input_length < max_input_length
        and 0 < label_length < max_label_length
    )


print("DATASET PREPARATION IN PROGRESS...")
raw_dataset = DatasetDict()
raw_dataset["train"] = load_all_datasets("train")
raw_dataset["eval"] = load_all_datasets("eval")

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["audio", "sentence"],
    num_proc=args.num_proc,
)

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 1) Build batch input_features from raw audio
        audios = [f["audio"] for f in features]
        samples = [a.get_all_samples().data[0] for a in audios]
        sampling_rate = 16000

        # text processing
        sentences: List[str] = [f["sentence"] for f in features]
        if do_lower_case:
            sentences = [s.lower() for s in sentences]
        if do_remove_punctuation and normalizer is not None:
            sentences = [normalizer(s).strip() for s in sentences]

        # Remove spaces from sentences if needed
        sentences = [s.replace(" ", "") for s in sentences]

        batch = self.processor(
            samples,
            language=args.language,
            text=sentences,
            punctuation=True,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            add_special_tokens=True,
        )

        batch["prompt_input_ids"] = batch["decoder_input_ids"].clone()
        batch["prompt_attention_mask"] = torch.ones_like(batch["prompt_input_ids"])
        batch["input_features"] = batch["input_features"].bfloat16()

        batch["labels"] = torch.cat(
            [batch["decoder_input_ids"], batch["labels"]], dim=-1
        )
        batch["decoder_input_ids"] = batch["labels"][:, :-1].clone()
        batch["labels"] = batch["labels"][:, 1:]
        prompt_length = batch["prompt_input_ids"].shape[1]
        if prompt_length > 1:
            batch["labels"][:, : prompt_length - 1] = -100
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
)
print("DATASET PREPARATION COMPLETED")


metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred).replace(" ", "") for pred in pred_str]
        label_str = [normalizer(label).replace(" ", "") for label in label_str]
        print(pred_str[0])
        print(label_str[0])

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


class PromptConditionedSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None,
        **gen_kwargs,
    ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        generation_inputs = {
            k: v
            for k, v in inputs.items()
            if k
            not in (
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
            )
        }
        if "prompt_input_ids" in inputs:
            generation_inputs["decoder_input_ids"] = inputs["prompt_input_ids"]
            generation_inputs["decoder_attention_mask"] = inputs[
                "prompt_attention_mask"
            ]

        with contextlib.nullcontext():
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        gen_config = self.model.generation_config
        default_gen_config = gen_config._get_default_generation_params()
        gen_config.update(**default_gen_config, defaults_only=True)

        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length
            )
        elif (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
            )

        loss_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("prompt_input_ids", "prompt_attention_mask")
        }
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**loss_inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, loss_inputs["labels"])
                        .detach()
                        .mean()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .detach()
                        .mean()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = loss_inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1
                )
        else:
            labels = None

        return loss, generated_tokens, labels


###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == "epoch":
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        lr_scheduler_type="linear",
        warmup_ratio=0.02,
        resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=16,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

elif args.train_strategy == "steps":
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        lr_scheduler_type="linear",
        warmup_ratio=0.02,
        resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=16,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

trainer = PromptConditionedSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

processor.save_pretrained(training_args.output_dir)

print("TRAINING IN PROGRESS...")
trainer.train()
print("DONE TRAINING")
