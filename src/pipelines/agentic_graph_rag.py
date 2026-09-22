import time
import re
from typing import List, Optional, Dict, Any
from src.pipelines.base import BasePipeline, PipelineResult
from src.graph.engine import GraphEngine, normalize_text
from src.vector.index import VectorIndex
from src.llm.client import LLMClient
from src.indicators.entropy import EpistemicEntropyTracker
from src.indicators.avi import AVITracker

class AgenticGraphRAGPipeline(BasePipeline):
    def __init__(self, graph_engine: Optional[GraphEngine] = None, 
                 vector_index: Optional[VectorIndex] = None,
                 llm_client: Optional[LLMClient] = None):
        super().__init__("Agentic_GraphRAG")
        self.graph_engine = graph_engine or GraphEngine()
        self.vector_index = vector_index or VectorIndex()
        self.vector_index.load()
        self.llm = llm_client or LLMClient()

    def answer_question(self, qid: str, question: str, qtype: str, 
                        gold_doc_ids: Optional[List[str]] = None) -> PipelineResult:
        start_time = time.time()
        
        # State Management
        state = {
            'qid': qid,
            'question': question,
            'qtype': qtype,
            'step': 0,
            'strategy': 'initial_triage',
            'strategy_changed': False,
            'evidence': {},
            'retrieved_doc_ids': set(),
            'citations': [],
            'intermediate_facts': {},
            'final_answer': None
        }

        entropy_tracker = EpistemicEntropyTracker(question, qtype)
        avi_tracker = AVITracker()
        trace: List[Dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Investigation Loop
        while state['step'] < 5 and state['final_answer'] is None:
            state['step'] += 1
            step_idx = state['step']
            step_start = time.time()

            tool_invoked = None
            tokens_in_step = 0

            # -------------------------------------------------------------
            # STEP 1: ORCHESTRATOR & ENTITY LINKING / TEMPORAL RESOLUTION
            # -------------------------------------------------------------
            if step_idx == 1:
                tool_invoked = "orchestrator_and_entity_linking"
                entities = self._extract_entities(question)
                state['evidence']['entities'] = entities

                # Handle Temporal Games resolution
                if qtype == 'temporal' or "immediately before" in question.lower() or "immediately after" in question.lower():
                    m_prev = re.search(r'(Summer|Winter)\s+Olympics\s+held\s+immediately\s+before\s+(\d{4})', question, re.I)
                    if m_prev:
                        season = m_prev.group(1).capitalize()
                        target_yr = int(m_prev.group(2))
                        prev_games = self.graph_engine.get_preceding_games(target_yr, season)
                        state['evidence']['resolved_games'] = prev_games
                        state['intermediate_facts']['target_year_games_resolved'] = bool(prev_games)

                if entities.get('venue'):
                    state['intermediate_facts']['venue_identified'] = True
                if entities.get('date'):
                    state['intermediate_facts']['date_identified'] = True
                if entities.get('sport'):
                    state['intermediate_facts']['sport_filtered'] = True

                tokens_in_step = 45 # Estimated orchestrator planning cost

            # -------------------------------------------------------------
            # STEP 2: DYNAMIC RETRIEVAL (GRAPH TRAVERSAL OR AGGREGATION)
            # -------------------------------------------------------------
            elif step_idx == 2:
                entities = state['evidence'].get('entities', {})

                # Type A: Aggregation
                if qtype == 'aggregation':
                    tool_invoked = "graph_aggregation_engine"
                    sport = entities.get('sport')
                    games = entities.get('games')
                    # extract threshold from question: "more than X competitors"
                    m_thresh = re.search(r'more than\s+(\d+)\s+competitors', question, re.I)
                    threshold = int(m_thresh.group(1)) if m_thresh else 0

                    events = self.graph_engine.find_events(sport=sport, games=games)
                    for ev in events:
                        state['retrieved_doc_ids'].add(ev['id'])
                    
                    count = self.graph_engine.aggregate_competitors(events, threshold)
                    state['evidence']['count'] = count
                    state['evidence']['matched_events_count'] = len(events)
                    state['intermediate_facts']['games_filtered'] = bool(games)
                    state['intermediate_facts']['competitor_threshold_applied'] = True
                    state['intermediate_facts']['entity_count_aggregated'] = True
                    state['final_answer'] = str(count)
                    tokens_in_step = 60

                # Type B: Superlative
                elif qtype == 'superlative':
                    tool_invoked = "graph_superlative_engine"
                    sport = entities.get('sport')
                    games = entities.get('games')
                    events = self.graph_engine.find_events(sport=sport, games=games)
                    for ev in events:
                        state['retrieved_doc_ids'].add(ev['id'])
                    
                    best_event = self.graph_engine.superlative_competitors(events, highest=True)
                    if best_event:
                        state['evidence']['best_event'] = best_event
                        state['retrieved_doc_ids'].add(best_event['id'])
                        state['intermediate_facts']['games_filtered'] = bool(games)
                        state['intermediate_facts']['competitor_ranked'] = True
                        state['intermediate_facts']['max_entity_selected'] = True
                        state['final_answer'] = best_event.get('title')
                    tokens_in_step = 60

                # Type C: Multi-Hop (Venue + Date -> Event -> Gold Winner)
                elif qtype == 'multi_hop':
                    tool_invoked = "multi_hop_graph_traversal"
                    venue = entities.get('venue')
                    date_str = entities.get('date')
                    events = self.graph_engine.find_event_by_venue_and_date(venue, date_str) if venue and date_str else []
                    
                    if not events and venue:
                        # Fallback: strategy change to venue-only + vector search
                        state['strategy_changed'] = True
                        events = self.graph_engine.find_events(venue=venue)
                    
                    if events:
                        ev = events[0]
                        state['evidence']['target_event'] = ev
                        state['retrieved_doc_ids'].add(ev['id'])
                        state['intermediate_facts']['event_resolved'] = True
                        gold = ev.get('gold_athlete')
                        if gold:
                            state['intermediate_facts']['gold_medalist_retrieved'] = True
                            state['final_answer'] = gold
                    tokens_in_step = 85

                # Type D: Temporal (Preceding Games -> Sport Event -> Winner)
                elif qtype == 'temporal':
                    tool_invoked = "temporal_graph_traversal"
                    resolved_games = state['evidence'].get('resolved_games') or entities.get('games')
                    sport = entities.get('sport')
                    
                    events = self.graph_engine.find_events(sport=sport, games=resolved_games)
                    q_norm = normalize_text(question)
                    q_words = set(q_norm.split())
                    stop_words = {'who', 'won', 'the', 'gold', 'medal', 'in', 'at', 'summer', 'winter', 
                                  'olympics', 'held', 'immediately', 'before', 'after', 'event'}
                    target_words = q_words - stop_words
                    
                    best_ev = None
                    best_score = -1.0
                    for ev in events:
                        title_norm = normalize_text(ev.get('title', ''))
                        title_words = set(title_norm.split())
                        score = float(len(target_words.intersection(title_words)))
                        # check weight / category match (e.g. '80 kg' vs '+80 kg')
                        for tw in target_words:
                            if len(tw) >= 3 and tw in title_norm:
                                score += 0.5
                        # Penalize '+' if not in query
                        if '+' in title_norm and '+' not in q_norm:
                            score -= 2.0
                        if score > best_score:
                            best_score = score
                            best_ev = ev

                    if best_ev:
                        state['evidence']['target_event'] = best_ev
                        state['retrieved_doc_ids'].add(best_ev['id'])
                        state['intermediate_facts']['discipline_event_matched'] = True
                        gold = best_ev.get('gold_athlete')
                        if gold:
                            state['intermediate_facts']['winner_retrieved'] = True
                            state['final_answer'] = gold
                    tokens_in_step = 90

                # Type E: Lookup
                else:
                    tool_invoked = "direct_entity_lookup"
                    matches = self.graph_engine.find_event_by_title_or_name(question)
                    if matches:
                        ev = matches[0]
                        state['evidence']['target_event'] = ev
                        state['retrieved_doc_ids'].add(ev['id'])
                        state['intermediate_facts']['event_matched'] = True
                        
                        # Check what metric is asked
                        if "nations" in question.lower():
                            state['final_answer'] = str(ev.get('nations', ''))
                            state['intermediate_facts']['metric_extracted'] = True
                        elif "competitors" in question.lower():
                            state['final_answer'] = str(ev.get('competitors', ''))
                            state['intermediate_facts']['metric_extracted'] = True
                        elif "gold" in question.lower():
                            state['final_answer'] = ev.get('gold_athlete', '')
                            state['intermediate_facts']['metric_extracted'] = True
                    tokens_in_step = 50

            # -------------------------------------------------------------
            # STEP 3: EVIDENCE GAP ANALYSIS & FALLBACK SYNTHESIS (IF NEEDED)
            # -------------------------------------------------------------
            elif step_idx == 3:
                # If still not answered, trigger Vector Retrieval + LLM Synthesis
                tool_invoked = "vector_search_and_llm_synthesis"
                top_chunks = self.vector_index.search(question, top_k=3)
                for c, _ in top_chunks:
                    state['retrieved_doc_ids'].add(c['doc_id'])
                
                context_str = "\n".join([f"[{c['doc_id']}]: {c['text']}" for c, _ in top_chunks])
                synth_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
                ans, tok_stats = self.llm.generate(synth_prompt)
                state['final_answer'] = ans
                tokens_in_step = tok_stats['total_tokens']
                total_prompt_tokens += tok_stats['prompt_tokens']
                total_completion_tokens += tok_stats['completion_tokens']

            # -------------------------------------------------------------
            # EVIDENCE EVALUATION & AVI COMPUTATION
            # -------------------------------------------------------------
            current_entropy = entropy_tracker.update_facets(state['intermediate_facts'])
            marginal_gain = entropy_tracker.marginal_gain
            step_latency = time.time() - step_start

            avi_record = avi_tracker.record_step(
                step_index=step_idx,
                tool_name=tool_invoked,
                marginal_gain=marginal_gain,
                current_entropy=current_entropy,
                tokens_used=tokens_in_step,
                latency_sec=step_latency
            )

            total_prompt_tokens += tokens_in_step
            trace.append({
                **avi_record,
                'strategy_changed': state['strategy_changed'],
                'evidence_keys': list(state['evidence'].keys()),
                'retrieved_ids_so_far': list(state['retrieved_doc_ids'])
            })

            # Check Optimal Stopping Frontier
            if avi_record['should_stop'] and state['final_answer'] is not None:
                break

        # Fallback if loop finishes without answer
        if not state['final_answer']:
            state['final_answer'] = "Unknown based on evidence"

        total_latency = time.time() - start_time
        total_tokens = total_prompt_tokens + total_completion_tokens

        return PipelineResult(
            pipeline_name=self.name,
            qid=qid,
            question=question,
            qtype=qtype,
            answer=state['final_answer'],
            gold_doc_ids=gold_doc_ids or [],
            retrieved_doc_ids=list(state['retrieved_doc_ids']),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            latency_sec=round(total_latency, 3),
            trace=trace,
            metadata={
                'average_avi': avi_tracker.average_avi,
                'final_entropy': entropy_tracker.current_entropy,
                'strategy_changed': state['strategy_changed'],
                'total_hops': state['step'],
                'stopping_reason': 'Optimal AVI Convergence' if entropy_tracker.current_entropy <= 0.05 else 'Max Hops Reached'
            }
        )

    def _extract_entities(self, question: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        m_games = re.search(r'\b(19\d\d|20\d\d)\s*(Summer|Winter)\b', question, re.I)
        if m_games:
            res['year'] = int(m_games.group(1))
            res['season'] = m_games.group(2).capitalize()
            res['games'] = f"{res['year']} {res['season']}"
        else:
            m_yr = re.search(r'\b(19\d\d|20\d\d)\b', question)
            if m_yr:
                res['year'] = int(m_yr.group(1))

        sports = ['athletics', 'biathlon', 'cycling', 'fencing', 'judo', 'rowing', 
                  'sailing', 'shooting', 'swimming', 'weightlifting', 'alpine skiing', 
                  'cross-country skiing', 'gymnastics', 'boxing', 'canoeing', 'speed skating', 
                  'wrestling', 'taekwondo', 'badminton', 'tennis']
        q_lower = question.lower()
        for s in sports:
            if s in q_lower:
                res['sport'] = s.title()
                break

        m_ven = re.search(r'held at\s+([A-Za-z0-9–\-\s,\'\.]+?)(?:\s+on\b|\s+during\b|\s+at\s+the\s+\d{4}|\?|$)', question, re.I)
        if m_ven:
            res['venue'] = m_ven.group(1).strip()

        m_date = re.search(r'\bon\s+([A-Za-z0-9–\-\s,]+?)(?:\s+at\b|\s+in\b|\?|$)', question, re.I)
        if m_date:
            res['date'] = m_date.group(1).strip()

        return res
