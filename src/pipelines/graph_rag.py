import time
import re
from typing import List, Optional, Dict, Any
from src.pipelines.base import BasePipeline, PipelineResult
from src.graph.engine import GraphEngine
from src.llm.client import LLMClient

class GraphRAGPipeline(BasePipeline):
    def __init__(self, graph_engine: Optional[GraphEngine] = None, llm_client: Optional[LLMClient] = None):
        super().__init__("GraphRAG")
        self.graph_engine = graph_engine or GraphEngine()
        self.llm = llm_client or LLMClient()

    def answer_question(self, qid: str, question: str, qtype: str, 
                        gold_doc_ids: Optional[List[str]] = None) -> PipelineResult:
        start_time = time.time()
        
        # 1. Extract potential entity references from question
        entities_found = self._extract_entities(question)
        
        # 2. Graph Traversal: 1-2 hop neighborhood retrieval
        matched_events: List[Dict[str, Any]] = []
        retrieved_ids: List[str] = []
        graph_triples = []

        # If question has title/event mentions
        event_matches = self.graph_engine.find_event_by_title_or_name(question)
        if event_matches:
            matched_events.extend(event_matches[:5])

        # If venue & date present
        if entities_found.get('venue') and entities_found.get('date'):
            vd_matches = self.graph_engine.find_event_by_venue_and_date(entities_found['venue'], entities_found['date'])
            matched_events.extend(vd_matches)

        # If sport & games present
        if entities_found.get('sport') or entities_found.get('games'):
            sg_matches = self.graph_engine.find_events(
                sport=entities_found.get('sport'),
                games=entities_found.get('games'),
                year=entities_found.get('year'),
                season=entities_found.get('season')
            )
            matched_events.extend(sg_matches[:10])

        # Deduplicate matched events
        seen = set()
        unique_events = []
        for ev in matched_events:
            eid = ev.get('id')
            if eid and eid not in seen:
                seen.add(eid)
                unique_events.append(ev)
                retrieved_ids.append(eid)
                # Build graph triples
                graph_triples.append(f"({ev['id']}) -[:HELD_AT]-> ({ev.get('venue')})")
                graph_triples.append(f"({ev['id']}) -[:IN_GAMES]-> ({ev.get('games')})")
                if ev.get('gold_athlete'):
                    graph_triples.append(f"({ev.get('gold_athlete')}) -[:WON_GOLD]-> ({ev['id']})")

        # 3. Build Graph-Augmented Context
        context_lines = ["=== Knowledge Graph Subgraph ==="]
        for tr in graph_triples[:15]:
            context_lines.append(f"  {tr}")
        
        context_lines.append("\n=== Supporting Event Records ===")
        for ev in unique_events[:5]:
            context_lines.append(
                f"Document [{ev['id']}]: {ev.get('title')}\n"
                f"  Venue: {ev.get('venue')} | Date: {ev.get('date_str')}\n"
                f"  Competitors: {ev.get('competitors')} | Nations: {ev.get('nations')}\n"
                f"  Gold Medalist: {ev.get('gold_athlete')} ({ev.get('gold_noc')})\n"
                f"  Snippet: {ev.get('text_snippet', '')[:250]}"
            )

        context_str = "\n".join(context_lines)
        system_prompt = (
            "You are a graph-augmented QA assistant answering strictly based on the provided Knowledge Graph context. "
            "If the answer is a person's name, return only the name. "
            "If the answer is a count or number, return only the number. "
            "If the answer is an event title, return the exact title."
        )
        user_prompt = f"Graph Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        # 4. LLM Generation
        answer, token_stats = self.llm.generate(user_prompt, system_instruction=system_prompt)
        latency = time.time() - start_time

        trace = [{
            'step': 1,
            'action': 'graph_traversal_and_synthesis',
            'extracted_entities': entities_found,
            'nodes_traversed': len(unique_events),
            'triples_retrieved': len(graph_triples),
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

    def _extract_entities(self, question: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        # Match year and season
        m_games = re.search(r'\b(19\d\d|20\d\d)\s*(Summer|Winter)\b', question, re.I)
        if m_games:
            res['year'] = int(m_games.group(1))
            res['season'] = m_games.group(2).capitalize()
            res['games'] = f"{res['year']} {res['season']}"
        else:
            m_yr = re.search(r'\b(19\d\d|20\d\d)\b', question)
            if m_yr:
                res['year'] = int(m_yr.group(1))

        # Match sports
        sports = ['athletics', 'biathlon', 'cycling', 'fencing', 'judo', 'rowing', 
                  'sailing', 'shooting', 'swimming', 'weightlifting', 'alpine skiing', 
                  'cross-country skiing', 'gymnastics', 'boxing', 'canoeing', 'speed skating', 'wrestling']
        q_lower = question.lower()
        for s in sports:
            if s in q_lower:
                res['sport'] = s.title()
                break

        # Match venue if "held at ..."
        m_ven = re.search(r'held at\s+([^,]+?)(?:\s+on\b|\s+at\b|\s+during\b|\?|$)', question, re.I)
        if m_ven:
            res['venue'] = m_ven.group(1).strip()

        # Match date if "on [date]"
        m_date = re.search(r'\bon\s+([A-Za-z0-9–\-\s,]+?)(?:\s+at\b|\s+in\b|\?|$)', question, re.I)
        if m_date:
            res['date'] = m_date.group(1).strip()

        return res
