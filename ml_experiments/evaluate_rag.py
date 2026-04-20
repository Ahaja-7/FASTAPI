import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_qwen_demo import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    ROOT,
    build_prompt,
    generate_answer,
    load_generator,
    load_jsonl,
    retrieve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=str(ROOT / "data" / "eval_questions.jsonl"))
    parser.add_argument("--docs-file", default=str(ROOT / "data" / "rag_docs.jsonl"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retrieval-only", action="store_true")
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def score_keywords(answer: str, must_include: list[str], must_not_include: list[str]) -> tuple[float, float]:
    normalized = answer.lower()
    include_hits = sum(1 for keyword in must_include if keyword.lower() in normalized)
    bad_hits = sum(1 for keyword in must_not_include if keyword.lower() in normalized)
    include_score = include_hits / len(must_include) if must_include else 1.0
    bad_keyword_rate = bad_hits / len(must_not_include) if must_not_include else 0.0
    return include_score, bad_keyword_rate


def retrieval_metrics(retrieved_ids: list[str], expected_ids: list[str], top_k: int) -> dict:
    expected = set(expected_ids)
    retrieved_top_k = retrieved_ids[:top_k]
    hits = [doc_id for doc_id in retrieved_top_k if doc_id in expected]

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_top_k, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": 1.0 if hits else 0.0,
        "recall": len(hits) / len(expected) if expected else 1.0,
        "precision": len(hits) / top_k if top_k else 0.0,
        "mrr": reciprocal_rank,
    }


def main() -> None:
    args = parse_args()
    eval_rows = load_jsonl(Path(args.eval_file))
    docs = load_jsonl(Path(args.docs_file))

    answer_embedder = None
    tokenizer = None
    model = None
    if not args.retrieval_only:
        answer_embedder = SentenceTransformer(args.embedding_model)
        tokenizer, model = load_generator(args.model_name, args.adapter_path)

    results = []
    for row in eval_rows:
        retrieved = retrieve(row["question"], docs, args.embedding_model, args.top_k)
        retrieved_ids = [doc["id"] for doc, _score in retrieved]
        metrics = retrieval_metrics(retrieved_ids, row["expected_doc_ids"], args.top_k)

        answer = ""
        answer_similarity = None
        semantic_distance = None
        include_score = None
        bad_keyword_rate = None

        if not args.retrieval_only:
            prompt = build_prompt(row["question"], retrieved)
            answer = generate_answer(tokenizer, model, prompt, args.max_new_tokens)
            embeddings = answer_embedder.encode([row["expected_answer"], answer], normalize_embeddings=True)
            answer_similarity = cosine_similarity(np.asarray(embeddings[0]), np.asarray(embeddings[1]))
            semantic_distance = 1.0 - answer_similarity
            include_score, bad_keyword_rate = score_keywords(
                answer,
                row.get("must_include", []),
                row.get("must_not_include", []),
            )

        result = {
            "id": row["id"],
            "question": row["question"],
            "retrieved_ids": retrieved_ids,
            **metrics,
            "answer_similarity": answer_similarity,
            "semantic_distance": semantic_distance,
            "include_score": include_score,
            "bad_keyword_rate": bad_keyword_rate,
            "answer": answer,
        }
        results.append(result)

        print(f"\n[{row['id']}] {row['question']}")
        print(f"retrieved: {', '.join(retrieved_ids)}")
        print(
            "retrieval "
            f"hit@{args.top_k}={metrics['hit']:.2f} "
            f"recall@{args.top_k}={metrics['recall']:.2f} "
            f"precision@{args.top_k}={metrics['precision']:.2f} "
            f"mrr={metrics['mrr']:.2f}"
        )
        if not args.retrieval_only:
            print(
                "answer "
                f"similarity={answer_similarity:.3f} "
                f"semantic_distance={semantic_distance:.3f} "
                f"include_score={include_score:.2f} "
                f"bad_keyword_rate={bad_keyword_rate:.2f}"
            )
            print(f"answer: {answer}")

    print("\nSummary")
    print(f"questions: {len(results)}")
    print(f"hit@{args.top_k}: {mean(item['hit'] for item in results):.3f}")
    print(f"recall@{args.top_k}: {mean(item['recall'] for item in results):.3f}")
    print(f"precision@{args.top_k}: {mean(item['precision'] for item in results):.3f}")
    print(f"mrr: {mean(item['mrr'] for item in results):.3f}")

    if not args.retrieval_only:
        print(f"answer_similarity: {mean(item['answer_similarity'] for item in results):.3f}")
        print(f"semantic_distance: {mean(item['semantic_distance'] for item in results):.3f}")
        print(f"include_score: {mean(item['include_score'] for item in results):.3f}")
        print(f"bad_keyword_rate: {mean(item['bad_keyword_rate'] for item in results):.3f}")


if __name__ == "__main__":
    main()
