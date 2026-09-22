import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.config import Config
from src.pipelines.vanilla_rag import VanillaRAGPipeline
from src.pipelines.graph_rag import GraphRAGPipeline
from src.pipelines.agentic_graph_rag import AgenticGraphRAGPipeline

def normalize_answer(ans: str) -> str:
    if not ans:
        return ""
    import re
    # Lowercase, strip punctuation and whitespace
    norm = ans.lower().strip()
    norm = re.sub(r'[^\w\s]', '', norm)
    norm = re.sub(r'\s+', ' ', norm)
    return norm

def is_match(pred: str, ground_truth: list) -> bool:
    if not ground_truth:
        return False
    norm_pred = normalize_answer(pred)
    for gt in ground_truth:
        norm_gt = normalize_answer(gt)
        if norm_pred == norm_gt or norm_gt in norm_pred or (len(norm_pred) > 3 and norm_pred in norm_gt):
            return True
    return False

def run_benchmark(max_questions: int = 100):
    Config.ensure_dirs()
    print("=" * 60)
    print("🚀 STARTING 3-WAY AGENTIC GRAPHRAG BENCHMARK")
    print(f"Dataset: {Config.PUBLIC_EVAL_FILE}")
    print("=" * 60)

    p1 = VanillaRAGPipeline()
    p2 = GraphRAGPipeline()
    p3 = AgenticGraphRAGPipeline()

    with open(Config.PUBLIC_EVAL_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()][:max_questions]

    results_p1 = []
    results_p2 = []
    results_p3 = []

    metrics = {
        'total_questions': len(questions),
        'by_pipeline': {
            'Vanilla_RAG': {'correct': 0, 'total_tokens': 0, 'total_latency': 0.0},
            'GraphRAG': {'correct': 0, 'total_tokens': 0, 'total_latency': 0.0},
            'Agentic_GraphRAG': {'correct': 0, 'total_tokens': 0, 'total_latency': 0.0, 'total_avi': 0.0}
        },
        'by_qtype': {}
    }

    start_all = time.time()

    for idx, q in enumerate(questions):
        qid = q['qid']
        text = q['question']
        qtype = q['qtype']
        gt = q.get('answer', [])

        if qtype not in metrics['by_qtype']:
            metrics['by_qtype'][qtype] = {
                'count': 0,
                'Vanilla_RAG_correct': 0,
                'GraphRAG_correct': 0,
                'Agentic_GraphRAG_correct': 0
            }
        metrics['by_qtype'][qtype]['count'] += 1

        # Run 3 pipelines
        r1 = p1.answer_question(qid, text, qtype, q.get('gold_doc_ids'))
        r2 = p2.answer_question(qid, text, qtype, q.get('gold_doc_ids'))
        r3 = p3.answer_question(qid, text, qtype, q.get('gold_doc_ids'))

        m1 = is_match(r1.answer, gt)
        m2 = is_match(r2.answer, gt)
        m3 = is_match(r3.answer, gt)

        if m1: metrics['by_pipeline']['Vanilla_RAG']['correct'] += 1; metrics['by_qtype'][qtype]['Vanilla_RAG_correct'] += 1
        if m2: metrics['by_pipeline']['GraphRAG']['correct'] += 1; metrics['by_qtype'][qtype]['GraphRAG_correct'] += 1
        if m3: metrics['by_pipeline']['Agentic_GraphRAG']['correct'] += 1; metrics['by_qtype'][qtype]['Agentic_GraphRAG_correct'] += 1

        metrics['by_pipeline']['Vanilla_RAG']['total_tokens'] += r1.total_tokens
        metrics['by_pipeline']['Vanilla_RAG']['total_latency'] += r1.latency_sec

        metrics['by_pipeline']['GraphRAG']['total_tokens'] += r2.total_tokens
        metrics['by_pipeline']['GraphRAG']['total_latency'] += r2.latency_sec

        metrics['by_pipeline']['Agentic_GraphRAG']['total_tokens'] += r3.total_tokens
        metrics['by_pipeline']['Agentic_GraphRAG']['total_latency'] += r3.latency_sec
        metrics['by_pipeline']['Agentic_GraphRAG']['total_avi'] += r3.metadata.get('average_avi', 0.0)

        results_p1.append(r1.model_dump())
        results_p2.append(r2.model_dump())
        results_p3.append(r3.model_dump())

        if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
            print(f"[{idx+1}/{len(questions)}] Processed | "
                  f"Agentic Acc: {metrics['by_pipeline']['Agentic_GraphRAG']['correct']/(idx+1)*100:.1f}% | "
                  f"GraphRAG Acc: {metrics['by_pipeline']['GraphRAG']['correct']/(idx+1)*100:.1f}% | "
                  f"Vanilla RAG Acc: {metrics['by_pipeline']['Vanilla_RAG']['correct']/(idx+1)*100:.1f}%")

    total_time = time.time() - start_all
    metrics['benchmark_runtime_sec'] = round(total_time, 2)

    # Compute averages
    n = len(questions)
    for p in ['Vanilla_RAG', 'GraphRAG', 'Agentic_GraphRAG']:
        metrics['by_pipeline'][p]['accuracy_pct'] = round(metrics['by_pipeline'][p]['correct'] / n * 100, 2)
        metrics['by_pipeline'][p]['avg_tokens'] = round(metrics['by_pipeline'][p]['total_tokens'] / n, 1)
        metrics['by_pipeline'][p]['avg_latency_sec'] = round(metrics['by_pipeline'][p]['total_latency'] / n, 3)

    metrics['by_pipeline']['Agentic_GraphRAG']['avg_avi'] = round(metrics['by_pipeline']['Agentic_GraphRAG']['total_avi'] / n, 3)

    # Save summary and detailed outputs
    summary_file = Config.RESULTS_DIR / "public_benchmark_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    detailed_file = Config.RESULTS_DIR / "public_benchmark_detailed.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump({
            'Vanilla_RAG': results_p1,
            'GraphRAG': results_p2,
            'Agentic_GraphRAG': results_p3
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("🏆 FINAL BENCHMARK RESULTS (100 QUESTIONS)")
    print("=" * 60)
    for p in ['Vanilla_RAG', 'GraphRAG', 'Agentic_GraphRAG']:
        print(f"  • {p:<18} | Accuracy: {metrics['by_pipeline'][p]['accuracy_pct']}% | "
              f"Avg Tokens: {metrics['by_pipeline'][p]['avg_tokens']} | "
              f"Avg Latency: {metrics['by_pipeline'][p]['avg_latency_sec']}s")
    print(f"  • Agentic Value Index (AVI): {metrics['by_pipeline']['Agentic_GraphRAG']['avg_avi']}")
    print("=" * 60)
    print(f"Summary saved to: {summary_file}")
    print(f"Detailed traces saved to: {detailed_file}")

if __name__ == '__main__':
    run_benchmark(100)
