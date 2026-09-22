# 🐯 **Agentic GraphRAG Hackathon Guidebook**

> Welcome, builders. This page has everything you need to navigate the Agentic GraphRAG Hackathon. Bookmark it, share it with your team, and use it as your single source of truth.


---


## About the Hackathon

RAG retrieves text. GraphRAG adds structure. But some questions need more - a system that can break the question apart, decide what to retrieve, evaluate what comes back, spot gaps or conflicts, and keep going until it has enough to answer.

The **Agentic GraphRAG Hackathon by TigerGraph** challenges you to build an AI agent that autonomously investigates complex questions using graph, vector, and document evidence. You'll benchmark three approaches side-by-side - RAG, GraphRAG, and Agentic GraphRAG - and prove exactly where agentic reasoning adds real value.

**The headline goal: figure out which questions need an agent, and which don't.** Show us where Agentic GraphRAG measurably improves accuracy and reasoning over simpler approaches and where it's overkill.


### Why Join


---


## ⚙️ Getting Started

You need two things before diving in: a **TigerGraph environment** and the **GraphRAG repo**.


### Step 1: Pick Your TigerGraph Environment

**Option A: TigerGraph Savanna (recommended).** Web-based, zero installation. Sign up at tgcloud.io - credits will be provided.

**Option B: Community Edition.** Free, runs locally. Download from dl.tigergraph.com.


### Step 2: Clone the GraphRAG Repo

bash

`Bash
git clone https://github.com/tigergraph/graphrag.git
`


### Step 3: Get an LLM API Key

You'll need access to any LLM provider of your choice for all three pipelines and for evaluation. Use whichever you're comfortable with. Most major providers offer free tiers that are more than enough for hackathon-scale usage.


---


## 🎯 Problem Statement


### The Problem

Some questions can be answered with a single retrieval step. Others require connecting entities, relationships, and evidence across multiple sources. The hardest questions require a system that can plan an investigation, decide what to retrieve, evaluate what comes back, identify gaps, perform additional steps, and decide when enough evidence exists to answer.

**RAG** retrieves similar text chunks. **GraphRAG** adds structure — entities, relationships, multi-hop reasoning. **Agentic GraphRAG** adds autonomous planning — the system decides its own retrieval path based on what it finds.

**The big question:** When does a complex question require an agentic, multi-step investigation rather than a single GraphRAG or RAG retrieval?


---


### How the Hackathon Works

This hackathon runs in two rounds:

**Round 1: Build Agentic GraphRAG (open to everyone)**

* Open to all registered participants globally
* Build a working Agentic GraphRAG system with the three-way benchmark (RAG, GraphRAG, Agentic GraphRAG)
* Your system should include an orchestrator agent, specialised retrieval/reasoning agents, and a benchmarking pipeline
* Top 15 teams advance to Round 2
**Round 2: Reasoning Over Time (Top 15 only)**

* **Window:** Sep 25 → Oct 1, 2026
* **Results announced:** Oct 7, 2026
* Extend your agent to reason over evolving, conflicting, and uncertain facts
* Your system must detect conflicting versions of a fact, determine what supersedes what, identify which sources are more authoritative, and handle uncertainty
* Finalists submit a demo video, writeup, and metrics dashboard
* Top teams present live to the judging panel

---


### What You'll Build

Three pipelines that answer the same questions, plus a comparison:

1. **Pipeline 1: RAG.** Retrieve relevant text through similarity search and generate an answer.
1. **Pipeline 2: GraphRAG.** Use graph structure, entities, relationships, and supporting content to retrieve context and answer.
1. **Pipeline 3: Agentic GraphRAG.** An agent plans the investigation, selects retrieval methods, evaluates intermediate results, and performs additional retrieval or reasoning steps as needed.
Your system should include:

* An **agent harness** to manage state, tools, context, evidence, and stopping criteria
* An **orchestrator agent** that determines what needs investigating and picks the next action — not a fixed retrieval sequence
* **Specialised agents** for: entity linking, graph traversal, similarity search, document retrieval, aggregation, multi-hop reasoning, evidence evaluation
The orchestrator's next move should depend on:

* The original question
* The graph and available entities
* Evidence returned from previous steps
* Information still needed to answer
For example, one question may need:

Entity Linking → Graph Traversal → Answer

Another may need:

Similarity Search → Identify Entity → Graph Traversal → Retrieve Supporting Documents → Answer

A more complex question may need several iterations before the system has enough evidence.


---


### Dataset

We provide a dataset with questions of increasing complexity. 
**Download it here: **[https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing](https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing)

The dataset includes:

**Corpus:** a document collection with the corpus name and source materials your system will ingest and reason over.

**100 evaluation questions:** these are yours to test, tune, and benchmark against. Run all three pipelines (RAG, GraphRAG, Agentic GraphRAG) on these and include the results in your metrics dashboard.

**50 hidden evaluation questions:** you won't see the answers for these. Run your system on all 50 and submit the raw outputs — tokens used, answers generated, and agentic trace. We use these to score your system against our held-out ground truth.

You're free to bring your own dataset too. The provided one is the common benchmark everyone is scored on, so use your own as a bonus to show your system handles real or messy data.


---


### Evaluation

For every question and every pipeline, measure:

**Accuracy:** correctness, completeness, grounding in available evidence

**Token Efficiency:** context tokens, LLM input tokens, LLM output tokens, total tokens per answer

**Trace & Agentic Behavior (for Agentic GraphRAG):**

* Number of retrieval and reasoning steps
* Retrieval methods selected
* Specialised agents invoked
* Tools called
* Time per operation
* Tokens per operation
* Total tokens used
* Number of chunks and citations
* Whether the system changed strategy during investigation
* When and why the system decided to stop
The objective is not simply to measure whether Agentic GraphRAG produces a better answer. It is to determine whether the additional reasoning and retrieval steps are worth the additional complexity and token cost.


---


### How You'll Be Judged


---


### Required Deliverables

**Round 1:**

1. Working Agentic GraphRAG system
1. GitHub repository
1. Architecture diagram
1. Demo video
1. Metrics dashboard comparing the three pipelines (tokens, accuracy, completeness)
1. Optional: social media post (tag @TigerGraph, counts in your favour)
**Round 2 (finalists):**

1. Refined Agentic GraphRAG system
1. Updated GitHub repository