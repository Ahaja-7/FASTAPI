import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

#파일 읽기
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SPAM_CSV = r"\\wsl.localhost\Ubuntu-22.04\home\jekal\data\spam.csv"
ROOT = Path(__file__).resolve().parent

#옵션 정의
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_SPAM_CSV)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "sms_qwen3_0_6b_lora"))
    parser.add_argument("--train-split-file", default=str(ROOT / "data" / "sms_spam_train.jsonl"))
    parser.add_argument("--test-split-file", default=str(ROOT / "data" / "sms_spam_test.jsonl"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()

#라벨 소문자로 정리, ham, spam이 아니면 에러처리
def normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label not in {"ham", "spam"}:
        raise ValueError(f"Unsupported label: {label!r}")
    return label

#csv파일 읽기, v1은 라벨, v2는 메시지
def load_sms_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="latin-1", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            label = normalize_label(row["v1"])
            message = row["v2"].strip()
            if message:
                rows.append({"label": label, "message": message})
    return rows

#훈련셋 테스트셋 나누기
def stratified_split(rows: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    train_rows = []
    test_rows = []
    for label_rows in by_label.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_size))
        test_rows.extend(shuffled[:test_count])
        train_rows.extend(shuffled[test_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows

#훈련/테스트 나눈 결과 jsonl 파일로 저장
def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

#모델에게 sms 보고 ham 또는 spam으로 답하라고 프롬프트 만들기
def build_prompt(message: str) -> str:
    return (
        "<|im_start|>system\n"
        "You classify SMS messages. Reply with exactly one label: ham or spam."
        "<|im_end|>\n"
        f"<|im_start|>user\n/no_think\nSMS:\n{message}\n\nLabel:"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

#프롬프트 뒤에 정답 라벨 붙이기(현재 코드에서 사용 x)
def build_training_text(row: dict) -> str:
    return f"{build_prompt(row['message'])}{row['label']}<|im_end|>"

#학습
def tokenize_for_training(tokenizer, row: dict, max_seq_length: int) -> dict:
    prompt = build_prompt(row["message"])
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

#dtype 선택
def get_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

#기본 모델 로딩
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

#LoRa 학습, 파인튜닝 담당
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

#LoRa 설정
    peft_config = LoraConfig(
        r=16, #LoRa rank, 클수록 표현력은 늘지만 파라미터 증가
        lora_alpha=32, #LoRa scaling값
        lora_dropout=0.05, #LoRa 경로에 dropout을 적용
        bias="none", #bias는 학습 안 함
        task_type="CAUSAL_LM", #causal language model용 LoRA입니다.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #attention projection과 MLP projection 계층에 LoRA를 붙임
    )

#주요 설정
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

#예측 결과 파싱
def parse_prediction(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\b(ham|spam)\b", text)
    return match.group(1) if match else "unknown"

#평가, 
def evaluate(args: argparse.Namespace, tokenizer, test_rows: list[dict]) -> None:
    import torch
    from peft import PeftModel

    model, _dtype = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()

    rows = test_rows[: args.eval_limit] if args.eval_limit else test_rows
    labels = ["ham", "spam"]
    confusion = {(actual, predicted): 0 for actual in labels for predicted in labels + ["unknown"]}
    predictions = []

    for row in rows:
        inputs = tokenizer(build_prompt(row["message"]), return_tensors="pt").to(model.device)
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
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        predicted = parse_prediction(generated_text)
        actual = row["label"]
        confusion[(actual, predicted)] = confusion.get((actual, predicted), 0) + 1
        predictions.append((actual, predicted))

    correct = sum(1 for actual, predicted in predictions if actual == predicted)
    total = len(predictions)
    accuracy = correct / total if total else 0.0

    print("\nEvaluation")
    print(f"test_samples: {total}")
    print(f"accuracy: {accuracy:.4f}")
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in labels if other != label)
        fn = sum(confusion[(label, other)] for other in labels + ["unknown"] if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    print("confusion:")
    for actual in labels:
        values = ", ".join(f"pred_{predicted}={confusion[(actual, predicted)]}" for predicted in labels + ["unknown"])
        print(f"  actual_{actual}: {values}")


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    train_split_file = Path(args.train_split_file)
    test_split_file = Path(args.test_split_file)

    rows = load_sms_rows(data_file)
    train_rows, test_rows = stratified_split(rows, args.test_size, args.seed)
    write_jsonl(train_split_file, train_rows)
    write_jsonl(test_split_file, test_rows)

    print(f"Loaded rows: {len(rows)} {dict(Counter(row['label'] for row in rows))}")
    print(f"Train rows: {len(train_rows)} {dict(Counter(row['label'] for row in train_rows))}")
    print(f"Test rows: {len(test_rows)} {dict(Counter(row['label'] for row in test_rows))}")
    print(f"Train split saved to: {train_split_file}")
    print(f"Test split saved to: {test_split_file}")

    if args.prepare_only:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir if args.skip_train else args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_train:
        train_lora(args, tokenizer, train_rows)

    evaluate(args, tokenizer, test_rows)


if __name__ == "__main__":
    main()
