import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
from src.config import Config
from src.graph.schema import OlympicEventNode, GraphEdge, KnowledgeGraphData

def clean_int(val: str) -> int | None:
    if not val:
        return None
    # Remove footnotes, extra text, commas
    m = re.search(r'\b(\d{1,3}(?:,\d{3})*|\d+)\b', str(val))
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except ValueError:
            return None
    return None

def parse_title(title: str):
    """Extract sport, year, season, and event name from title if matching standard format."""
    # Pattern: [Sport] at the [Year] [Summer/Winter] Olympics – [Event Name]
    m = re.search(r'^([A-Za-z\s\-]+?)\s+at the\s+(\d{4})\s+(Summer|Winter)\s+Olympics\s*[–—-]\s*(.+)$', title)
    if m:
        return {
            'sport': m.group(1).strip(),
            'year': int(m.group(2)),
            'season': m.group(3).strip(),
            'event_name': m.group(4).strip()
        }
    return None

def build_knowledge_graph(corpus_file: Path = Config.CORPUS_FILE) -> KnowledgeGraphData:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[GraphEdge] = []

    # Track unique entities
    venues = set()
    games_set = set()
    sports_set = set()
    athletes_set = set()

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc.get('doc_id')
            title = doc.get('title', '')
            text = doc.get('text', '')

            # Parse infobox fields
            info: Dict[str, str] = {}
            m = re.search(r'\[Infobox Olympic event\](.*?)(?:\n\n|\n[A-Z]|$)', text, re.DOTALL)
            if m:
                box_text = m.group(1)
                for bline in box_text.split('\n'):
                    parts = bline.strip().split(':', 1)
                    if len(parts) == 2:
                        k = parts[0].strip().lower()
                        v = parts[1].strip()
                        info[k] = v

            # Fallback metadata from title
            title_meta = parse_title(title)
            sport = title_meta['sport'] if title_meta else ""
            year = title_meta['year'] if title_meta else None
            season = title_meta['season'] if title_meta else ""
            event_name = info.get('event') or (title_meta['event_name'] if title_meta else title)

            # Games resolution
            games_str = info.get('games', '')
            if games_str:
                gm = re.search(r'(\d{4})\s*(Summer|Winter)', games_str, re.I)
                if gm:
                    year = int(gm.group(1))
                    season = gm.group(2).capitalize()
                    games_str = f"{year} {season}"
            elif year and season:
                games_str = f"{year} {season}"

            date_val = info.get('date') or info.get('dates') or ""
            venue_val = info.get('venue') or info.get('venues') or ""
            competitors_val = clean_int(info.get('competitors', ''))
            nations_val = clean_int(info.get('nations', ''))
            gold_athlete = info.get('gold', '')
            gold_noc = info.get('goldnoc', '')
            silver_athlete = info.get('silver', '')
            silver_noc = info.get('silvernoc', '')
            bronze_athlete = info.get('bronze', '')
            bronze_noc = info.get('bronzenoc', '')
            prev_yr = clean_int(info.get('prev', ''))
            next_yr = clean_int(info.get('next', ''))

            # First 500 chars of text after infobox as snippet
            snippet = text[:600] if not m else text[m.end():m.end()+600].strip()

            event_node = OlympicEventNode(
                id=doc_id,
                title=title,
                event_name=event_name,
                sport=sport,
                games=games_str,
                year=year,
                season=season,
                venue=venue_val,
                date_str=date_val,
                competitors=competitors_val,
                nations=nations_val,
                gold_athlete=gold_athlete,
                gold_noc=gold_noc,
                silver_athlete=silver_athlete,
                silver_noc=silver_noc,
                bronze_athlete=bronze_athlete,
                bronze_noc=bronze_noc,
                prev_year=prev_yr,
                next_year=next_yr,
                url=doc.get('url', ''),
                text_snippet=snippet
            )

            nodes[doc_id] = {
                'type': 'OlympicEvent',
                **event_node.model_dump()
            }

            # Relational Edges
            if venue_val:
                venue_id = f"venue:{venue_val.lower().strip()}"
                if venue_id not in nodes:
                    nodes[venue_id] = {'type': 'Venue', 'name': venue_val}
                edges.append(GraphEdge(source=doc_id, target=venue_id, rel_type='HELD_AT'))

            if games_str:
                games_id = f"games:{games_str.lower().strip()}"
                if games_id not in nodes:
                    nodes[games_id] = {'type': 'Games', 'name': games_str, 'year': year, 'season': season}
                edges.append(GraphEdge(source=doc_id, target=games_id, rel_type='IN_GAMES'))

            if sport:
                sport_id = f"sport:{sport.lower().strip()}"
                if sport_id not in nodes:
                    nodes[sport_id] = {'type': 'Sport', 'name': sport}
                edges.append(GraphEdge(source=doc_id, target=sport_id, rel_type='OF_SPORT'))

            if gold_athlete:
                athlete_id = f"athlete:{gold_athlete.lower().strip()}"
                if athlete_id not in nodes:
                    nodes[athlete_id] = {'type': 'Athlete', 'name': gold_athlete, 'noc': gold_noc}
                edges.append(GraphEdge(source=athlete_id, target=doc_id, rel_type='WON_GOLD', properties={'noc': gold_noc}))

    # Link temporal Games sequence
    # Summer sequence: 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020
    # Winter sequence: 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022
    summer_years = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
    winter_years = [1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

    for yrs, season in [(summer_years, "Summer"), (winter_years, "Winter")]:
        for i in range(1, len(yrs)):
            curr_id = f"games:{yrs[i]} {season.lower()}"
            prev_id = f"games:{yrs[i-1]} {season.lower()}"
            if curr_id in nodes and prev_id in nodes:
                edges.append(GraphEdge(source=curr_id, target=prev_id, rel_type='PRECEDED_BY'))
                edges.append(GraphEdge(source=prev_id, target=curr_id, rel_type='SUCCEEDED_BY'))

    return KnowledgeGraphData(nodes=nodes, edges=edges)

def save_graph(kg: KnowledgeGraphData, out_dir: Path = Config.PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = out_dir / "graph_nodes.json"
    edges_file = out_dir / "graph_edges.json"

    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(kg.nodes, f, ensure_ascii=False, indent=2)
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump([e.model_dump() for e in kg.edges], f, ensure_ascii=False, indent=2)

    print(f"Saved {len(kg.nodes)} nodes to {nodes_file}")
    print(f"Saved {len(kg.edges)} edges to {edges_file}")

if __name__ == "__main__":
    kg = build_knowledge_graph()
    save_graph(kg)
