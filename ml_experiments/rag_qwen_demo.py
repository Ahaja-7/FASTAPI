import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def retrieve(question: str, docs: list[dict], embedding_model: str, top_k: int) -> list[tuple[dict, float]]:
    embedder = SentenceTransformer(embedding_model)
    doc_embeddings = embedder.encode([doc["text"] for doc in docs], normalize_embeddings=True)
    query_embedding = embedder.encode([question], normalize_embeddings=True)

    vectors = np.asarray(doc_embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    scores, indexes = index.search(np.asarray(query_embedding, dtype="float32"), top_k)
    return [(docs[int(idx)], float(score)) for idx, score in zip(indexes[0], scores[0]) if idx >= 0]


def load_generator(model_name: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def build_prompt(question: str, retrieved_docs: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(
        f"[문서 {idx}] id={doc['id']}, score={score:.4f}\n{doc['text']}"
        for idx, (doc, score) in enumerate(retrieved_docs, start=1)
    )
    return (
        "<|im_start|>system\n"
        "너는 한국어로 답변하는 RAG 분석 도우미다. "
        "반드시 제공된 문서만 근거로 답하고, 근거가 없으면 모른다고 답한다."
        "<|im_end|>\n"
        f"<|im_start|>user\n질문:\n{question}\n\n검색 문서:\n{context}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--docs-file", default=str(ROOT / "data" / "rag_docs.jsonl"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only print retrieved documents without loading Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_jsonl(Path(args.docs_file))
    retrieved_docs = retrieve(args.question, docs, args.embedding_model, args.top_k)

    print("Retrieved documents:")
    for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        print(f"{idx}. {doc['id']} score={score:.4f} {doc['text']}")

    if args.retrieval_only:
        return

    tokenizer, model = load_generator(args.model_name, args.adapter_path)
    prompt = build_prompt(args.question, retrieved_docs)
    answer = generate_answer(tokenizer, model, prompt, args.max_new_tokens)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
