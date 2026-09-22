import time
from typing import List, Optional
from src.pipelines.base import BasePipeline, PipelineResult
from src.vector.index import VectorIndex
from src.llm.client import LLMClient

class VanillaRAGPipeline(BasePipeline):
    def __init__(self, vector_index: Optional[VectorIndex] = None, llm_client: Optional[LLMClient] = None):
        super().__init__("Vanilla_RAG")
        self.vector_index = vector_index or VectorIndex()
        self.vector_index.load()
        self.llm = llm_client or LLMClient()

    def answer_question(self, qid: str, question: str, qtype: str, 
                        gold_doc_ids: Optional[List[str]] = None) -> PipelineResult:
        start_time = time.time()
        
        # 1. Similarity search
        top_chunks = self.vector_index.search(question, top_k=5)
        retrieved_ids = [c[0]['doc_id'] for c in top_chunks]

        # 2. Build prompt
        context_parts = []
        for i, (chunk, score) in enumerate(top_chunks):
            context_parts.append(f"--- Document {i+1} [{chunk['doc_id']}]: {chunk['title']} ---\n{chunk['text']}")
        context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are a factual QA assistant answering strictly based on the provided documents. "
            "If the answer is a person's name, return only the name. "
            "If the answer is a count or number, return only the number. "
            "If the answer is an event title, return the exact title."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        # 3. LLM Generation
        answer, token_stats = self.llm.generate(user_prompt, system_instruction=system_prompt)
        latency = time.time() - start_time

        trace = [{
            'step': 1,
            'action': 'vector_search_and_generate',
            'retrieved_docs': retrieved_ids,
            'top_similarity_scores': [round(c[1], 4) for c in top_chunks],
            'tokens': token_stats['total_tokens'],
            'latency': round(latency, 3)
        }]

        return PipelineResult(
            pipeline_name=self.name,
            qid=qid,
            question=question,
            qtype=qtype,
            answer=answer,
            gold_doc_ids=gold_doc_ids or [],
            retrieved_doc_ids=retrieved_ids,
            prompt_tokens=token_stats['prompt_tokens'],
            completion_tokens=token_stats['completion_tokens'],
            total_tokens=token_stats['total_tokens'],
            latency_sec=round(latency, 3),
            trace=trace
        )
