from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OlympicEventNode(BaseModel):
    id: str = Field(..., description="Wikidata QID / Document ID (e.g., Q303623)")
    title: str = Field(..., description="Full document title")
    event_name: str = Field("", description="Specific event name")
    sport: str = Field("", description="Sport or discipline (e.g., Athletics, Biathlon)")
    games: str = Field("", description="Games identifier (e.g., 2012 Summer, 2018 Winter)")
    year: Optional[int] = Field(None, description="Year of the games (e.g., 2012)")
    season: str = Field("", description="Summer or Winter")
    venue: str = Field("", description="Venue where the event was held")
    date_str: str = Field("", description="Date string from infobox")
    competitors: Optional[int] = Field(None, description="Number of competitors")
    nations: Optional[int] = Field(None, description="Number of participating nations")
    gold_athlete: str = Field("", description="Gold medal winner(s)")
    gold_noc: str = Field("", description="NOC code of gold medal winner")
    silver_athlete: str = Field("", description="Silver medal winner(s)")
    silver_noc: str = Field("", description="NOC code of silver medal winner")
    bronze_athlete: str = Field("", description="Bronze medal winner(s)")
    bronze_noc: str = Field("", description="NOC code of bronze medal winner")
    prev_year: Optional[int] = Field(None, description="Previous Olympic year for event")
    next_year: Optional[int] = Field(None, description="Next Olympic year for event")
    url: str = Field("", description="Source Wikipedia URL")
    text_snippet: str = Field("", description="Summary snippet of document")

class GraphEdge(BaseModel):
    source: str
    target: str
    rel_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeGraphData(BaseModel):
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)
