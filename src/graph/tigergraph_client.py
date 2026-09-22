import logging
from typing import Optional, Dict, Any, List
from src.config import Config

logger = logging.getLogger(__name__)

GSQL_SCHEMA = """
// TigerGraph GSQL Schema Definition for Agentic GraphRAG
CREATE GRAPH OlympicGraph()

USE GRAPH OlympicGraph

CREATE VERTEX OlympicEvent (
    PRIMARY_ID doc_id STRING,
    title STRING,
    event_name STRING,
    sport STRING,
    games STRING,
    year INT,
    season STRING,
    venue STRING,
    date_str STRING,
    competitors INT,
    nations INT,
    gold_athlete STRING,
    gold_noc STRING,
    silver_athlete STRING,
    silver_noc STRING,
    bronze_athlete STRING,
    bronze_noc STRING,
    url STRING
) WITH STATS="OUTDEGREE_BY_EDGETYPE"

CREATE VERTEX Venue (PRIMARY_ID name STRING)
CREATE VERTEX Games (PRIMARY_ID name STRING, year INT, season STRING)
CREATE VERTEX Sport (PRIMARY_ID name STRING)
CREATE VERTEX Athlete (PRIMARY_ID name STRING, noc STRING)

CREATE UNDIRECTED EDGE HELD_AT (FROM OlympicEvent, TO Venue)
CREATE UNDIRECTED EDGE IN_GAMES (FROM OlympicEvent, TO Games)
CREATE UNDIRECTED EDGE OF_SPORT (FROM OlympicEvent, TO Sport)
CREATE DIRECTED EDGE WON_GOLD (FROM Athlete, TO OlympicEvent, noc STRING)
CREATE DIRECTED EDGE PRECEDED_BY (FROM Games, TO Games)

INSTALL SCHEMA
"""

class TigerGraphClient:
    def __init__(self, host: str = Config.TIGERGRAPH_HOST, 
                 username: str = Config.TIGERGRAPH_USERNAME,
                 password: str = Config.TIGERGRAPH_PASSWORD,
                 graphname: str = Config.TIGERGRAPH_GRAPHNAME,
                 secret: str = Config.TIGERGRAPH_SECRET,
                 api_token: str = Config.TIGERGRAPH_API_TOKEN):
        self.host = host
        self.username = username
        self.password = password
        self.graphname = graphname
        self.secret = secret
        self.api_token = api_token
        self.conn = None
        self._connected = False

    def connect(self) -> bool:
        if not self.host:
            logger.info("No TigerGraph host provided; running in local in-memory graph mode.")
            return False
        try:
            import pyTigerGraph as tg
            self.conn = tg.TigerGraphConnection(
                host=self.host,
                username=self.username,
                password=self.password,
                graphname=self.graphname,
                apiToken=self.api_token or None
            )
            if self.secret and not self.api_token:
                self.api_token = self.conn.getToken(self.secret)
                self.conn.apiToken = self.api_token
            self._connected = True
            logger.info(f"Successfully connected to TigerGraph at {self.host}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to TigerGraph ({e}); falling back to local graph engine.")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    def get_gsql_schema_script(self) -> str:
        return GSQL_SCHEMA.strip()
