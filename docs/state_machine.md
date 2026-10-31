
# Resume Matching Agent - State Machine

```mermaid
flowchart TD
    A([START]) --> B[Classify User Intent]

    B --> C{Intent}

    C -->|New Job Search| D[Parse Job Description]
    D --> E[Extract Requirements]
    E --> F[Search Resumes using ChromaDB]
    F --> G[Rank Candidates]

    C -->|Refine Requirements| H[Update Requirements]
    H --> F

    G --> I[Deep Screen Top 10 Candidates]
    I --> J[Generate Final Recommendations]
    J --> K[Human Feedback / Recruiter Review]
    K --> L[Generate Matching Report]
    L --> M([END])

    C -->|Compare Candidates| N[Compare Candidates]
    C -->|Explain Ranking| O[Explain Ranking]
    C -->|Generate Questions| P[Generate Interview Questions]
    C -->|General Question| Q[Answer General Question]

    N --> M
    O --> M
    P --> M
    Q --> M
```

## Workflow Description

1. **Classify User Intent:** Identifies whether the recruiter wants a new search, requirement refinement, comparison, explanation, or interview questions.
2. **Parse Job Description:** Captures the job description supplied by the recruiter.
3. **Extract Requirements:** Identifies required skills, must-have skills, and minimum experience.
4. **Search Resumes:** Retrieves relevant resume chunks from ChromaDB.
5. **Rank Candidates:** Scores candidates using semantic similarity and skill coverage.
6. **Deep Screening:** Performs a second-round analysis of shortlisted candidates.
7. **Final Recommendations:** Generates interview, further-review, or do-not-recommend suggestions.
8. **Human Feedback:** Pauses the workflow for recruiter review and captures feedback.
9. **Generate Report:** Produces the final candidate matching report.
10. **Conversation Routes:** Supports candidate comparisons, ranking explanations, interview questions, and general recruitment questions.
