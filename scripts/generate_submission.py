import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.config import Config
from src.pipelines.vanilla_rag import VanillaRAGPipeline
from src.pipelines.graph_rag import GraphRAGPipeline
from src.pipelines.agentic_graph_rag import AgenticGraphRAGPipeline

def generate_hidden_submission():
    Config.ensure_dirs()
    print("=" * 60)
    print("🚀 GENERATING SUBMISSION FOR 50 HIDDEN EVALUATION QUESTIONS")
    print(f"Input: {Config.HIDDEN_EVAL_FILE}")
    print("=" * 60)

    p1 = VanillaRAGPipeline()
    p2 = GraphRAGPipeline()
    p3 = AgenticGraphRAGPipeline()

    with open(Config.HIDDEN_EVAL_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]

    submission_records = []
    out_file = Config.RESULTS_DIR / "eval_hidden_submission.jsonl"

    for idx, q in enumerate(questions):
        qid = q['qid']
        text = q['question']
        qtype = q['qtype']

        # Run 3 pipelines
        r1 = p1.answer_question(qid, text, qtype)
        r2 = p2.answer_question(qid, text, qtype)
        r3 = p3.answer_question(qid, text, qtype)

        record = {
            'qid': qid,
            'question': text,
            'qtype': qtype,
            'answers': {
                'vanilla_rag': r1.answer,
                'graph_rag': r2.answer,
                'agentic_graph_rag': r3.answer
            },
            'tokens': {
                'vanilla_rag_tokens': r1.total_tokens,
                'graph_rag_tokens': r2.total_tokens,
                'agentic_graph_rag_tokens': r3.total_tokens
            },
            'latency_sec': {
                'vanilla_rag_latency': r1.latency_sec,
                'graph_rag_latency': r2.latency_sec,
                'agentic_graph_rag_latency': r3.latency_sec
            },
            'agentic_trace': r3.trace,
            'agentic_metadata': r3.metadata
        }

        submission_records.append(record)

    with open(out_file, 'w', encoding='utf-8') as f:
        for rec in submission_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n✅ Successfully generated submission for {len(submission_records)} hidden questions!")
    print(f"Output saved to: {out_file}")

if __name__ == '__main__':
    generate_hidden_submission()
