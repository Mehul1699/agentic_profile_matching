# Resume Matching Agent

A LangGraph-based conversational AI agent for resume screening, candidate matching, and job requirement analysis.

## Overview

This project extends an existing Resume RAG (Retrieval-Augmented Generation) and Profile Matching system into a conversational agent. It aims to help users search resumes, rank candidates against job requirements, compare candidate profiles, and understand the reasoning behind matching results.

## Objectives

- Build a stateful agent workflow using LangGraph.
- Extract must-have and nice-to-have requirements from job descriptions.
- Reuse existing resume retrieval and candidate-ranking functionality.
- Support conversational queries and iterative requirement refinement.
- Compare candidates and generate interview screening questions.
- Produce explainable candidate reports highlighting strengths and skill gaps.
- Implement multi-round candidate screening.

## Planned Technology Stack

- **Python** — core implementation
- **LangGraph** — agent workflow and state management
- **ChromaDB** — vector storage and semantic retrieval
- **OpenAI-compatible API via OpenRouter** — embedding generation and potential LLM capabilities
- **CLI** — initial conversational interface

## Project Structure

```text
resume-matching-agent/
├── matching_agent.py
├── job_matcher.py
├── resume_rag.py
├── fs_tools.py
├── app.py
├── resumes/
├── job_descriptions/
├── tests/
├── docs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*The structure above represents the planned project layout; files and features are being implemented incrementally.*

## Planned Features

### 1. Agent Architecture
- Stateful LangGraph workflow
- Job requirement extraction
- Resume retrieval and candidate ranking
- Report generation and human feedback

### 2. Conversational Interface
- Natural-language candidate searches
- Side-by-side candidate comparisons
- Ranking explanations
- Dynamic refinement of job requirements

### 3. Advanced Screening
- Initial candidate screening
- In-depth analysis of shortlisted candidates
- Evidence-based hire/no-hire recommendations
- Explainable match reports and interview questions

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the required API key in a local `.env` file, using `.env.example` as a template.
5. Run the application after implementation is complete.

## Security and Data Privacy

- Do not commit API keys, `.env` files, or other secrets.
- Use only resumes and candidate data that you are authorized to process.
- Keep generated databases and temporary outputs out of version control unless explicitly required.

## Project Status

**In development.**

The project is being built incrementally on top of the existing Resume RAG and Profile Matching system. Agent orchestration, conversational interaction, testing, and documentation will be added as implementation progresses.

## Deliverables

- LangGraph-based agent implementation
- State-machine diagram
- Interactive chat interface
- At least five conversation-flow test scenarios
- Demo video demonstrating the agent's capabilities
