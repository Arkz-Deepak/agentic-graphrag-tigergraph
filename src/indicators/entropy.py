from typing import Dict, Any, List, Set

class EpistemicEntropyTracker:
    """
    Tracks the collapse of ambiguity and information entropy across agentic reasoning steps.
    """
    def __init__(self, question: str, qtype: str):
        self.question = question
        self.qtype = qtype
        self.required_facets = self._determine_facets()
        self.fulfilled_facets: Set[str] = set()
        self.history: List[float] = [1.0] # Starts at max entropy (1.0)

    def _determine_facets(self) -> Set[str]:
        facets = {"domain_relevance"}
        q = self.question.lower()
        if self.qtype == 'multi_hop':
            facets.update({"venue_identified", "date_identified", "event_resolved", "gold_medalist_retrieved"})
        elif self.qtype == 'temporal':
            facets.update({"target_year_games_resolved", "discipline_event_matched", "winner_retrieved"})
        elif self.qtype == 'aggregation':
            facets.update({"sport_filtered", "games_filtered", "competitor_threshold_applied", "entity_count_aggregated"})
        elif self.qtype == 'superlative':
            facets.update({"sport_filtered", "games_filtered", "competitor_ranked", "max_entity_selected"})
        else: # lookup
            facets.update({"event_matched", "metric_extracted"})
        return facets

    def update_facets(self, new_facts: Dict[str, Any]) -> float:
        for f in self.required_facets:
            if f in new_facts and new_facts[f]:
                self.fulfilled_facets.add(f)
        
        completeness = len(self.fulfilled_facets) / max(1, len(self.required_facets))
        current_entropy = max(0.0, 1.0 - completeness)
        self.history.append(current_entropy)
        return current_entropy

    @property
    def current_entropy(self) -> float:
        return self.history[-1]

    @property
    def marginal_gain(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return max(0.0, self.history[-2] - self.history[-1])
