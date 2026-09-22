import json
import re
from typing import List, Dict, Any, Optional
import networkx as nx
from src.config import Config

def normalize_text(text: str) -> str:
    if not text:
        return ""
    import unicodedata
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r'[–—−-]', '-', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class GraphEngine:
    def __init__(self, nodes_file=Config.PROCESSED_DIR / "graph_nodes.json", 
                 edges_file=Config.PROCESSED_DIR / "graph_edges.json"):
        with open(nodes_file, 'r', encoding='utf-8') as f:
            self.nodes: Dict[str, Dict[str, Any]] = json.load(f)
        with open(edges_file, 'r', encoding='utf-8') as f:
            self.edges: List[Dict[str, Any]] = json.load(f)

        self.graph = nx.MultiDiGraph()
        for nid, data in self.nodes.items():
            self.graph.add_node(nid, **data)
        for e in self.edges:
            self.graph.add_edge(e['source'], e['target'], key=e['rel_type'], **e.get('properties', {}))

        # Indexing for sub-millisecond retrieval
        self._events_by_games = {}
        self._events_by_sport = {}
        self._events_by_venue = {}
        self._build_indices()

    def _build_indices(self):
        for nid, data in self.nodes.items():
            if data.get('type') == 'OlympicEvent':
                # by games
                gms = data.get('games', '').lower()
                if gms:
                    self._events_by_games.setdefault(gms, []).append(nid)
                
                # by sport
                spt = data.get('sport', '').lower()
                if spt:
                    self._events_by_sport.setdefault(spt, []).append(nid)

                # by venue
                ven = data.get('venue', '').lower()
                if ven:
                    self._events_by_venue.setdefault(ven, []).append(nid)

    def find_event_by_title_or_name(self, query: str) -> List[Dict[str, Any]]:
        query_norm = normalize_text(query)
        results = []
        for nid, data in self.nodes.items():
            if data.get('type') == 'OlympicEvent':
                title_norm = normalize_text(data.get('title', ''))
                event_norm = normalize_text(data.get('event_name', ''))
                if (title_norm and (query_norm == title_norm or (len(title_norm) >= 12 and title_norm in query_norm) or (len(query_norm) >= 12 and query_norm in title_norm))):
                    results.append(data)
                elif (event_norm and (query_norm == event_norm or (len(event_norm) >= 12 and event_norm in query_norm))):
                    results.append(data)
        return results

    def find_events(self, sport: Optional[str] = None, games: Optional[str] = None, 
                    year: Optional[int] = None, season: Optional[str] = None,
                    venue: Optional[str] = None) -> List[Dict[str, Any]]:
        candidate_ids = None

        if games:
            g_key = games.lower().strip()
            # fuzzy or substring match on games
            matching = [nids for g, nids in self._events_by_games.items() if g_key in g or g in g_key]
            candidate_ids = set([item for sublist in matching for item in sublist])

        if sport:
            s_key = sport.lower().strip()
            matching = [nids for s, nids in self._events_by_sport.items() if s_key in s or s in s_key]
            s_set = set([item for sublist in matching for item in sublist])
            candidate_ids = s_set if candidate_ids is None else candidate_ids.intersection(s_set)

        if venue:
            v_key = venue.lower().strip()
            matching = [nids for v, nids in self._events_by_venue.items() if v_key in v or v in v_key]
            v_set = set([item for sublist in matching for item in sublist])
            candidate_ids = v_set if candidate_ids is None else candidate_ids.intersection(v_set)

        if candidate_ids is None:
            candidate_ids = [nid for nid, d in self.nodes.items() if d.get('type') == 'OlympicEvent']

        results = []
        for nid in candidate_ids:
            d = self.nodes[nid]
            if year and d.get('year') != year:
                continue
            if season and d.get('season', '').lower() != season.lower():
                continue
            results.append(d)
        return results

    def get_preceding_games(self, year: int, season: str) -> Optional[str]:
        games_id = f"games:{year} {season.lower()}"
        if games_id in self.graph:
            for _, target, key in self.graph.out_edges(games_id, keys=True):
                if key == 'PRECEDED_BY':
                    return self.graph.nodes[target].get('name')
        return None

    def find_event_by_venue_and_date(self, venue_query: str, date_query: str) -> List[Dict[str, Any]]:
        v_norm = venue_query.lower().strip()
        d_norm = date_query.lower().strip()

        candidates = []
        for nid, data in self.nodes.items():
            if data.get('type') == 'OlympicEvent':
                node_venue = data.get('venue', '').lower().strip()
                node_date = data.get('date_str', '').lower().strip()

                if node_venue and (v_norm in node_venue or node_venue in v_norm):
                    if self._date_matches(d_norm, node_date):
                        # compute overlap score
                        score = 2.0 if v_norm == node_venue else 1.0
                        if d_norm == node_date:
                            score += 2.0
                        candidates.append((data, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def _date_matches(self, query_date: str, node_date: str) -> bool:
        if not node_date:
            return False
        if query_date in node_date or node_date in query_date:
            return True
        # Match day and month
        months = ['january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
        q_m = [m for m in months if m in query_date]
        n_m = [m for m in months if m in node_date]
        if q_m and n_m and set(q_m).intersection(set(n_m)):
            # Check numbers (days)
            q_days = re.findall(r'\b\d{1,2}\b', query_date)
            n_days = re.findall(r'\b\d{1,2}\b', node_date)
            if set(q_days).intersection(set(n_days)):
                return True
        return False

    def aggregate_competitors(self, events: List[Dict[str, Any]], min_competitors: int) -> int:
        return sum(1 for e in events if e.get('competitors') is not None and e['competitors'] > min_competitors)

    def superlative_competitors(self, events: List[Dict[str, Any]], highest: bool = True) -> Optional[Dict[str, Any]]:
        valid = [e for e in events if e.get('competitors') is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x['competitors']) if highest else min(valid, key=lambda x: x['competitors'])
