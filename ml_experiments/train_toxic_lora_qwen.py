import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TRAIN_CSV = r"\\wsl.localhost\Ubuntu-22.04\home\jekal\data\train.csv"
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "toxic_qwen3_0_6b_lora"))
    parser.add_argument("--train-split-file", default=str(ROOT / "data" / "toxic_train.jsonl"))
    parser.add_argument("--eval-split-file", default=str(ROOT / "data" / "toxic_eval.jsonl"))
    parser.add_argument("--eval-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def resolve_wsl_path(path: str) -> Path:
    if path.startswith(r"\\wsl.localhost\Ubuntu-22.04"):
        return Path(path.replace(r"\\wsl.localhost\Ubuntu-22.04", "").replace("\\", "/"))
    return Path(path)


def normalize_label(value: str) -> str:
    value = str(value).strip()
    if value == "1":
        return "toxic"
    if value == "0":
        return "non-toxic"
    raise ValueError(f"Unsupported toxic label: {value!r}")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            comment = row["comment_text"].strip()
            if comment:
                rows.append({"comment_text": comment, "label": normalize_label(row["toxic"])})
    return rows


def stratified_split(rows: list[dict], eval_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    train_rows = []
    eval_rows = []
    for label_rows in by_label.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        eval_count = max(1, round(len(shuffled) * eval_size))
        eval_rows.extend(shuffled[:eval_count])
        train_rows.extend(shuffled[eval_count:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(comment: str) -> str:
    return (
        "<|im_start|>system\n"
        "Classify whether the comment is toxic. Reply with exactly one label: toxic or non-toxic."
        "<|im_end|>\n"
        f"<|im_start|>user\n/no_think\nComment:\n{comment}\n\nLabel:"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def tokenize_for_training(tokenizer, row: dict, max_seq_length: int) -> dict:
    prompt = build_prompt(row["comment_text"])
    full_text = f"{prompt}{row['label']}<|im_end|>"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    labels = tokenized["input_ids"].copy()
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized


def get_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = get_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, dtype


def train_lora(args: argparse.Namespace, tokenizer, train_rows: list[dict]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, dtype = load_base_model(args.model_name)
    train_dataset = Dataset.from_list(train_rows)
    tokenized_dataset = train_dataset.map(
        lambda row: tokenize_for_training(tokenizer, row, args.max_seq_length),
        remove_columns=train_dataset.column_names,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch" if args.max_steps == -1 else "steps",
        save_steps=100,
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")


def parse_prediction(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if re.search(r"\bnon[- ]?toxic\b", text):
        return "non-toxic"
    if re.search(r"\btoxic\b", text):
        return "toxic"
    return "unknown"


def evaluate(args: argparse.Namespace, tokenizer, eval_rows: list[dict]) -> None:
    import torch
    from peft import PeftModel

    model, _dtype = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()

    rows = eval_rows[: args.eval_limit] if args.eval_limit else eval_rows
    labels = ["non-toxic", "toxic"]
    confusion = {(actual, predicted): 0 for actual in labels for predicted in labels + ["unknown"]}
    predictions = []

    for row in rows:
        inputs = tokenizer(build_prompt(row["comment_text"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        predicted = parse_prediction(tokenizer.decode(generated, skip_special_tokens=True))
        actual = row["label"]
        confusion[(actual, predicted)] = confusion.get((actual, predicted), 0) + 1
        predictions.append((actual, predicted))

    correct = sum(1 for actual, predicted in predictions if actual == predicted)
    total = len(predictions)
    print("\nEvaluation")
    print(f"eval_samples: {total}")
    print(f"accuracy: {correct / total if total else 0.0:.4f}")
    print("confusion:")
    for actual in labels:
        values = ", ".join(f"pred_{predicted}={confusion[(actual, predicted)]}" for predicted in labels + ["unknown"])
        print(f"  actual_{actual}: {values}")


def main() -> None:
    args = parse_args()
    data_file = resolve_wsl_path(args.data_file)
    train_split_file = Path(args.train_split_file)
    eval_split_file = Path(args.eval_split_file)

    rows = load_rows(data_file)
    train_rows, eval_rows = stratified_split(rows, args.eval_size, args.seed)
    write_jsonl(train_split_file, train_rows)
    write_jsonl(eval_split_file, eval_rows)

    print(f"Loaded rows: {len(rows)} {dict(Counter(row['label'] for row in rows))}")
    print(f"Train rows: {len(train_rows)} {dict(Counter(row['label'] for row in train_rows))}")
    print(f"Eval rows: {len(eval_rows)} {dict(Counter(row['label'] for row in eval_rows))}")
    print(f"Train split saved to: {train_split_file}")
    print(f"Eval split saved to: {eval_split_file}")

    if args.prepare_only:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir if args.skip_train else args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_train:
        train_lora(args, tokenizer, train_rows)

    evaluate(args, tokenizer, eval_rows)


if __name__ == "__main__":
    main()
