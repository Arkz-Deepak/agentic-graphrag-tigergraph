from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PipelineResult(BaseModel):
    pipeline_name: str
    qid: str
    question: str
    qtype: str
    answer: str
    gold_doc_ids: List[str] = Field(default_factory=list)
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BasePipeline(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def answer_question(self, qid: str, question: str, qtype: str, 
                        gold_doc_ids: Optional[List[str]] = None) -> PipelineResult:
        pass
