from typing import List, Dict, Any
from src.config import Config

class AVITracker:
    """
    Agentic Value Index (AVI) Tracker:
    Calculates the marginal information gain per token and determines optimal stopping criteria.
    """
    def __init__(self, lambda_latency: float = Config.AVI_LATENCY_WEIGHT, 
                 epsilon_stop: float = Config.AVI_EPSILON):
        self.lambda_latency = lambda_latency
        self.epsilon_stop = epsilon_stop
        self.step_records: List[Dict[str, Any]] = []

    def record_step(self, step_index: int, tool_name: str, marginal_gain: float, 
                    current_entropy: float, tokens_used: int, latency_sec: float) -> Dict[str, Any]:
        """
        Calculates AVI for the step.
        """
        denom = max(1, tokens_used) * (1.0 + self.lambda_latency * latency_sec)
        # Scaled AVI score
        avi_score = (marginal_gain / denom) * 1000.0 * (1.0 - current_entropy + 0.1)

        record = {
            'step': step_index,
            'tool': tool_name,
            'marginal_gain': round(marginal_gain, 4),
            'entropy': round(current_entropy, 4),
            'tokens': tokens_used,
            'latency': round(latency_sec, 3),
            'avi': round(avi_score, 4),
            'should_stop': (current_entropy <= 0.05) or (step_index >= 2 and avi_score < self.epsilon_stop)
        }
        self.step_records.append(record)
        return record

    @property
    def total_tokens(self) -> int:
        return sum(r['tokens'] for r in self.step_records)

    @property
    def total_latency(self) -> float:
        return sum(r['latency'] for r in self.step_records)

    @property
    def average_avi(self) -> float:
        if not self.step_records:
            return 0.0
        return sum(r['avi'] for r in self.step_records) / len(self.step_records)
