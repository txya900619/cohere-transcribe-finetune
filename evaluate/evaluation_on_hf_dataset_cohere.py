import argparse
import os
import re
from typing import Union

import torch
from datasets import (
    Audio,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from jiwer import process_characters
from jiwer.alignment import _construct_comparison_string
from jiwer.process import CharacterOutput, WordOutput
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

import evaluate

cer_metric = evaluate.load("cer")


def visualize_alignment(
    output: Union[WordOutput, CharacterOutput],
    show_measures: bool = True,
    skip_correct: bool = True,
    ids: list[str] = None,
) -> str:
    references = output.references
    hypothesis = output.hypotheses
    alignment = output.alignments
    is_cer = isinstance(output, CharacterOutput)

    final_str = ""
    for gt, hp, chunks, sample_id in zip(references, hypothesis, alignment, ids):
        if skip_correct and len(chunks) == 1 and chunks[0].type == "equal":
            continue

        final_str += f"sentence {sample_id.replace('.wav', '')}\n"
        final_str += _construct_comparison_string(
            gt, hp, chunks, include_space_seperator=not is_cer
        )
        final_str += "\n"

    if show_measures:
        final_str += f"number of sentences: {len(alignment)}\n"
        final_str += f"substitutions={output.substitutions} "
        final_str += f"deletions={output.deletions} "
        final_str += f"insertions={output.insertions} "
        final_str += f"hits={output.hits}\n"

        if is_cer:
            final_str += f"\ncer={output.cer * 100:.2f}%\n"
        else:
            final_str += f"\nmer={output.mer * 100:.2f}%"
            final_str += f"\nwil={output.wil * 100:.2f}%"
            final_str += f"\nwip={output.wip * 100:.2f}%"
            final_str += f"\nwer={output.wer * 100:.2f}%\n"
    else:
        final_str = final_str[:-1]

    return final_str


def get_text_column_name(column_names):
    for name in (
        "hanzi",
        "text",
        "sentence",
        "normalized_text",
        "transcript",
        "transcription",
    ):
        if name in column_names:
            return name
    raise ValueError(f"Unable to find transcript column in {column_names}")


def normalize_text(text):
    text = re.sub(r"[\.\?\!\;\,\，\。\？\！]", "", text)
    text = re.sub(r"\s+", "", text).strip().lower()
    return text


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.hf_model)
    model = CohereAsrForConditionalGeneration.from_pretrained(args.hf_model).to(device)
    model.eval()

    config_names = args.configs or get_dataset_config_names(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    for config in config_names:
        split_names = get_dataset_split_names(args.dataset, config)
        if args.split not in split_names:
            print(f"skip {config}: no {args.split} split")
            continue

        dataset = load_dataset(args.dataset, config, split=args.split)
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        text_column = get_text_column_name(dataset.column_names)
        dataset = dataset.filter(lambda x: x["duration"] < args.max_duration)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

        predictions = []
        references = []
        norm_predictions = []
        norm_references = []

        def map_to_pred(batch):
            audios = batch["audio"]
            samples = []
            batch_sampling_rate = None
            for audio in audios:
                audio_samples = audio.get_all_samples()
                waveform = audio_samples.data
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                else:
                    waveform = waveform[0]
                samples.append(waveform)
                if batch_sampling_rate is None:
                    batch_sampling_rate = audio_samples.sample_rate

            inputs = processor(
                samples,
                language=args.language,
                punctuation=True,
                sampling_rate=batch_sampling_rate,
                return_tensors="pt",
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)

            with torch.no_grad():
                predicted_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )

            transcriptions = processor.decode(
                predicted_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=args.language,
            )
            if isinstance(transcriptions, str):
                transcriptions = [transcriptions]

            predictions.extend(transcriptions)
            references.extend(batch[text_column])
            norm_predictions.extend([normalize_text(x) for x in transcriptions])
            norm_references.extend([normalize_text(x) for x in batch[text_column]])
            return {}

        dataset.map(map_to_pred, batched=True, batch_size=args.batch_size)

        cer = round(
            100 * cer_metric.compute(references=references, predictions=predictions), 2
        )
        norm_cer = round(
            100
            * cer_metric.compute(
                references=norm_references, predictions=norm_predictions
            ),
            2,
        )

        print(f"{config} CER: {cer}")
        print(f"{config} NORMALIZED CER: {norm_cer}")

        output_path = os.path.join(
            args.output_dir,
            f"{args.dataset.replace('/', '_')}_{config}_{args.hf_model.replace('/', '_')}.txt",
        )
        with open(output_path, "w") as f:
            f.write(f"CER: {cer}\n")
            f.write(f"NORMALIZED CER: {norm_cer}\n\n\n")

            if args.log_normalized:
                out = process_characters(norm_references, norm_predictions)
            else:
                out = process_characters(references, predictions)

            f.write(
                visualize_alignment(
                    out,
                    show_measures=True,
                    skip_correct=False,
                    ids=dataset["id"],
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model", type=str, default="CohereLabs/cohere-transcribe-03-2026"
    )
    parser.add_argument(
        "--dataset", type=str, default="formospeech/hakkaradio_news_clean"
    )
    parser.add_argument("--configs", type=str, nargs="*", default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="predictions_dir")
    parser.add_argument(
        "--log_normalized",
        required=False,
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    main(parser.parse_args())
