
# Agentic Resume and Profile Matching System

An AI-powered resume screening and candidate matching system built using Python, LangGraph, ChromaDB, and OpenRouter.

The system helps recruiters identify relevant candidates from a resume database, rank candidates against job requirements, compare profiles, explain ranking decisions, and generate interview questions.

## Key Features

- **Job Requirement Extraction:** Extracts required skills, must-have skills, and minimum experience from job descriptions.
- **Semantic Resume Search:** Uses embeddings and ChromaDB to retrieve relevant resume content.
- **Candidate Ranking:** Ranks candidates using semantic similarity and skill coverage.
- **Requirement Refinement:** Allows recruiters to update job requirements during a conversation and rerun matching.
- **Candidate Comparison:** Compares shortlisted candidates based on scores, matched skills, and missing skills.
- **Explainable Matching:** Provides reasons behind candidate rankings.
- **Multi-Round Screening:** Performs initial ranking followed by deeper analysis of shortlisted candidates.
- **Interview Question Generation:** Generates candidate-specific interview questions.
- **Human-in-the-Loop Review:** Pauses for recruiter feedback before producing the final report.

## Technology Stack

- Python 3.12
- LangGraph
- LangChain Core
- OpenAI-compatible API through OpenRouter
- ChromaDB
- Sentence embeddings: `openai/text-embedding-3-small`
- Chat model: `openai/gpt-4o-mini`
- PyPDF
- python-dotenv

## Project Structure

```text
agentic_profile_matching/
├── matching_agent.py
├── job_matcher.py
├── resume_rag.py
├── fs_tools.py
├── resumes/
├── job_descriptions/
├── chroma_db/
├── tests/
│   └── test_scenarios.md
├── docs/
│   └── state_machine.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create and activate a virtual environment

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

Add your OpenRouter API key and model:

```dotenv
OPENAI_API_KEY=your_openrouter_api_key
CHAT_MODEL=openai/gpt-4o-mini
```

Keep your actual `.env` file private. Never commit API keys to GitHub.

### 5. Add resumes

Place PDF or supported resume files inside the `resumes/` folder.

### 6. Index resumes

Run:

```bash
python resume_rag.py
```

This processes the resumes and stores their embeddings in ChromaDB.

### 7. Start the agent

Run:

```bash
python matching_agent.py
```

Follow the prompts in the terminal to interact with the recruiter assistant.

## Example Conversation

**Recruiter:** Find React developers with at least 3 years of experience.

**Agent:** Extracts requirements, searches resumes, ranks candidates, performs deeper screening, and generates recommendations.

**Recruiter:** Also require AWS.

**Agent:** Updates the requirements and reruns candidate matching.

**Recruiter:** Compare the top 3 candidates.

**Agent:** Presents a comparison of the shortlisted candidates.

**Recruiter:** Why is the first candidate ranked higher?

**Agent:** Explains the ranking using match scores and resume evidence.

**Recruiter:** Generate interview questions for the top candidate.

**Agent:** Generates relevant interview questions.

The agent also pauses for human review and captures recruiter feedback before producing the final report.

## Candidate Matching Method

Candidate ranking uses two main components:

- Semantic similarity: 45%
- Skill coverage: 55%

The resulting score is a ranking aid, not a probability of hiring success.

Must-have skills and minimum experience are also considered when determining candidate eligibility.

## Multi-Round Screening

**Round 1 — Initial Screening**
- Retrieves relevant resume content.
- Calculates candidate match scores.
- Checks must-have skills and experience.
- Creates a shortlist of up to 10 candidates.

**Round 2 — Deep Screening**
- Analyzes shortlisted candidates in greater detail.
- Identifies strengths, missing skills, relevant experience, and potential concerns.
- Generates final recommendations.

**Human Review**
- Pauses for recruiter feedback.
- Records the feedback in the final report.

## Testing

Conversation-flow test scenarios are documented in:

`tests/test_scenarios.md`

These cover new searches, requirement refinement, candidate comparison, ranking explanations, interview questions, and human feedback.

## State Machine Architecture

The LangGraph workflow is documented in:

`docs/state_machine.md`

## Important Notes

- Matching scores support recruiter decision-making and should not be treated as hiring probabilities.
- Missing information in retrieved resume excerpts does not necessarily mean a candidate lacks that skill.
- Recruiters should review original resumes and verify important qualifications before making hiring decisions.
- Final hiring decisions remain with human recruiters.

## Future Improvements

- Streamlit or Gradio graphical interface.
- Parallel candidate screening.
- More advanced feedback-driven recommendation updates.
- Improved experience calculation using employment dates.
- Persistent conversation storage.
