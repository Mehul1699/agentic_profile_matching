from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import re

from langgraph.graph import StateGraph, START, END

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

import job_matcher as jm

import os
from dotenv import load_dotenv
from openai import OpenAI

import json

import traceback

load_dotenv()

chat_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1"
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    job_description: str
    job_requirements: dict
    candidates: list[dict]
    shortlist: list[dict]
    reasoning: dict
    report: str
    intent: str
    intent_details: dict
    follow_up_query: str
    final_recommendations: list[dict]
    human_feedback: str


def parse_job(state: AgentState) -> dict:
    latest_message = state["messages"][-1].content

    return {
        "job_description": latest_message,
        "report": "",
        "candidates": [],
        "shortlist": [],
        "reasoning": {},
        "job_requirements": {},
    }


def extract_requirments(state: AgentState) -> dict:
    jd = state["job_description"]

    skills = jm.extract_job_skills(jd)
    must_have = jm.extract_must_have_skills(jd, skills)

    min_experience = 0

    experience_patterns = [
        r"\b(?:at least|minimum of|min\.?)\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|professional\s+|work\s+)?experience\b",
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+of\s+experience\b",
    ]

    for pattern in experience_patterns:
        match = re.search(pattern, jd, re.IGNORECASE)

        if match:
            min_experience = float(match.group(1))
            break

    return {
        "job_requirements": {
            "skills": skills,
            "must_have_skills": must_have,
            "min_years_experience": min_experience,
        }
    }


def search_resumes(state: AgentState) -> dict:
    jd = state.get("job_description", "")
    requirements = state.get("job_requirements", {})

    # Include updated requirements in the search query
    skills = requirements.get("skills", [])
    min_experience = requirements.get("min_years_experience", 0)

    search_query = jd

    if skills:
        search_query += "\nRequired skills: " + ", ".join(skills)

    if min_experience:
        search_query += f"\nMinimum experience: {min_experience} years"

    if jm.collection.count() == 0:
        raise ValueError("The resume database is empty. " "Run resume_rag.py first.")

    response = jm.client.embeddings.create(
        model=jm.EMBEDDING_MODEL, input=[search_query]
    )

    embedding = response.data[0].embedding

    count = min(200, jm.collection.count())

    results = jm.collection.query(
        query_embeddings=[embedding],
        n_results=count,
        include=["documents", "metadatas", "distances"],
    )

    candidates = jm.build_candidate_profile(results)

    return {"candidates": list(candidates.values())}


def rank_candidates(state: AgentState) -> dict:
    requirements = state["job_requirements"]
    ranked = []

    minimum_years = requirements.get("min_years_experience", 0)

    for candidate in state["candidates"]:
        required_skills = requirements.get("skills", [])
        score_data = jm.score_candidate(candidate, required_skills)

        resume_text = candidate["all_text"]

        missing_must_haves = [
            skill
            for skill in requirements.get("must_have_skills", [])
            if not jm.contains_skill(resume_text, skill)
        ]

        experience_status = check_experience(resume_text, minimum_years)

        if missing_must_haves or experience_status == "Insufficient":
            eligibility = "Not eligible"
        elif experience_status == "Unknown":
            eligibility = "Review"
        else:
            eligibility = "Eligible"

        explanation = jm.make_explanation(
            candidate, score_data["matched_skills"], score_data["missing_skills"]
        )

        explanation += f" Experience check: {experience_status}."

        if missing_must_haves:
            explanation += (
                " Missing required skills: " + ", ".join(missing_must_haves) + "."
            )

        ranked.append(
            {
                "candidate_name": candidate["name"],
                "resume_path": candidate["path"],
                "resume_text": resume_text,
                "match_score": score_data["match_score"],
                "matched_skills": score_data["matched_skills"],
                "missing_skills": score_data["missing_skills"],
                "missing_must_have_skills": missing_must_haves,
                "experience_status": experience_status,
                "minimum_years_experience": minimum_years,
                "eligibility": eligibility,
                "explanation": explanation,
            }
        )

    ranked.sort(key=lambda candidate: candidate["match_score"], reverse=True)

    eligible_candidates = [
        candidate for candidate in ranked if candidate["eligibility"] == "Eligible"
    ]

    shortlist = eligible_candidates[:10] if eligible_candidates else ranked[:10]

    return {
        "candidates": ranked,
        "shortlist": shortlist,
        "reasoning": {
            candidate["candidate_name"]: candidate["explanation"]
            for candidate in ranked
        },
    }


def check_experience(resume_text: str, minimum_years: int) -> str:
    if minimum_years <= 0:
        return "Meets"

    pattern = (
        r"(\d+(?:\.d+)?)\s*\+?\s"
        r"(?:years?|yrs?)\s+(?:of\s+)?"
        r"(?:professional\s+|work\s+)?experience"
    )

    matches = re.findall(pattern, resume_text, re.IGNORECASE)

    if not matches:
        return "Unknown"

    stated_years = max(float(years) for years in matches)

    if stated_years >= minimum_years:
        return "Matches"

    return "Insufficient"


def generate_report(state: AgentState) -> dict:
    requirements = state["job_requirements"]
    candidates = state["shortlist"]
    human_feedback = state.get("human_feedback", "")

    lines = [
        "Candidate Matching Report",
        "=" * 30,
        f"Skills requested: {', '.join(requirements['skills']) or 'None detected'}",
        f"Minimum experience requested: {requirements['min_years_experience']} years",
        "",
        "ROUND 1: INITIAL RANKING",
        "-" * 40,
    ]

    if not candidates:
        lines.append("No candidates were retrieved.")

    else:
        recommendations = {
            item["candidate_name"]: item
            for item in state.get("final_recommendations", [])
        }

        for rank, candidate in enumerate(candidates, start=1):
            deep_analysis = candidate.get(
                "deep_analysis", "Deep screening analysis not available."
            )

            if isinstance(deep_analysis, list):
                deep_analysis = "\n".join(str(item) for item in deep_analysis)
            elif not isinstance(deep_analysis, str):
                deep_analysis = str(deep_analysis)

            final_result = recommendations.get(candidate["candidate_name"], {})

            recommendation = final_result.get("recommendation", "Not available")

            recommendation_reasoning = final_result.get(
                "reasoning", "No final reasoning available."
            )

            lines.extend(
                [
                    f"{rank}. {candidate['candidate_name']}",
                    f"    Match score: {candidate['match_score']}/100",
                    f"    Eligibility: {candidate['eligibility']}",
                    f"    Matched skills: {', '.join(candidate['matched_skills']) or 'None found'}",
                    f"    Missing skills: {', '.join(candidate['missing_skills']) or 'None detected'}",
                    f"    Explanation: {candidate['explanation']}",
                    "",
                    "     ROUND 2: DEEP SCREENING ANALYSIS",
                    "     " + "-" * 35,
                    deep_analysis,
                    "",
                    f"    FINAL RECOMMENDATION: {recommendation}",
                    f"    Reasoning: {recommendation_reasoning}",
                ]
            )

    lines.extend(
        [
            "=" * 40,
            "IMPORTANT NOTES",
            "-" * 40,
            "Scores are ranking aids, not probabilities.",
            "Eligibility and screening findings should be verified against original resumes.",
            "Missing information in a resume does not necessarily mean the candidate lacks that skill.",
            "The report supports human decision-making; it does not make the final hiring decision.",
        ]
    )

    lines.extend(
        [
            "",
            "HUMAN REVIEW",
            "-" * 40,
            human_feedback if human_feedback else "No recruiter feedback provided.",
        ]
    )

    report = "\n".join(lines)
    print(report)

    return {"report": report}


def classify_user_intent(
    latest_message: str,
    conversation_history: list,
    current_requirements: dict,
    has_candidates: bool,
) -> dict:

    allowed_intents = [
        "new_search",
        "refine_requirements",
        "compare_candidates",
        "explain_ranking",
        "generate_questions",
        "general_question",
    ]

    recent_history = conversation_history[-8:]

    system_prompt = """
                        You are the intent classifier for a resume-matching agent.

                        Determine the user's intent using their latest message,
                        the conversation history, current requirements, and whether
                        candidate results are available.

                        Choose exactly one intent:
                        - new_search: start a new candidate search.
                        - refine_requirements: modify the existing search criteria.
                        - compare_candidates: compare candidates from existing results.
                        - explain_ranking: explain a ranking, score, or candidate decision.
                        - generate_questions: generate interview questions for a candidate.
                        - general_question: ask a general question or something that
                        does not fit the other categories.

                        Rules:
                        1. Use conversation history to interpret short follow-up messages.
                        2. A request to change existing criteria is refine_requirements.
                        3. A request to start a different job search is new_search.
                        4. Do not invent requirements the user did not specify.
                        5. If no candidates exist yet, do not assume previous search
                        results are available.
                        6. Preserve existing criteria unless the user asks to change them.
                        7. Extract only changes explicitly requested by the user.
                        8. If the intent is unclear, choose general_question.



                        INTENT CLASSIFICATION RULES:

                        A. compare_candidates
                        Choose this ONLY when the user explicitly asks to compare
                        two or more candidates.

                        Examples:
                        - Compare the top three candidates.
                        - Compare John and Jane.
                        - Who has more React experience, John or Jane?

                        B. explain_ranking
                        Choose this when the user asks WHY one candidate ranked
                        higher or received a better score than another.

                        Examples:
                        - What makes the first candidate better than the second?
                        - Why is John ranked higher than Jane?
                        - Why did this candidate get 85 instead of 70?
                        - Explain the ranking.

                        IMPORTANT:
                        Questions containing phrases such as "why", "what makes",
                        "ranked higher", "better than", or "higher score" should
                        normally be classified as explain_ranking when asking
                        about ranking reasons.

                        Never classify these examples as general_question.



                        Return a JSON object with exactly these fields:
                        {
                        "intent": "one allowed intent",
                        "candidate_count": 3,
                        "candidate_names": [],
                        "updated_requirements": {
                            "skills_to_add": [],
                            "skills_to_remove": [],
                            "min_years_experience": null,
                            "remove_experience_requirement": false
                        },
                        "question": "brief description of what the user wants"
                        }

                        Use candidate_count = 3 when a comparison asks for three candidates.
                        Otherwise use null if the number is not specified.
                        Use empty lists for unspecified skill changes.
                        Use null for an unspecified experience change.
                        Use false for remove_experience_requirement unless the user
                        explicitly removes that requirement.
"""

    context = {
        "latest_message": latest_message,
        "conversation_history": recent_history,
        "current_requirements": current_requirements,
        "candidate_results_available": has_candidates,
    }

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)

    ranking_phrases = [
        "why",
        "what makes",
        "ranked higher",
        "higher score",
        "better than",
        "explain the ranking",
    ]

    message_lower = latest_message.lower()

    if any(phrase in message_lower for phrase in ranking_phrases):
        result["intent"] = "explain_ranking"

    if result.get("intent") not in allowed_intents:
        result["intent"] = "general_question"

    return result


def classify_intent(state: AgentState) -> dict:
    latest_message = state["messages"][-1].content

    result = classify_user_intent(
        latest_message=latest_message,
        conversation_history=[
            {"role": msg.type, "content": msg.content} for msg in state["messages"][:-1]
        ],
        current_requirements=state.get("job_requirements", {}),
        has_candidates=bool(state.get("candidates")),
    )

    return {
        "intent": result["intent"],
        "intent_details": result,
        "follow_up_query": latest_message,
    }


def route_by_intent(state: AgentState) -> str:
    intent = state["intent"]

    routes = {
        "new_search": "parse_job",
        "refine_requirements": "refine_requirements",
        "compare_candidates": "compare_candidates",
        "explain_ranking": "explain_ranking",
        "generate_questions": "generate_questions",
        "general_question": "generate_question",
    }

    return routes.get(intent, "general_question")


def refine_requirements(state: AgentState) -> dict:
    current_requirements = state.get("job_requirements", {}).copy()

    changes = state["intent_details"]["updated_requirements"]

    skills = current_requirements.get("skills", []).copy()
    must_have_skills = current_requirements.get("must_have_skills", []).copy()

    # Add newly requested skills
    for skill in changes.get("skills_to_add", []):
        if skill.lower() not in [s.lower() for s in skills]:
            skills.append(skill)

        if skill.lower() not in [s.lower() for s in must_have_skills]:
            must_have_skills.append(skill)

    # Remove skills that user no longer wants
    for skill in changes.get("skills_to_remove", []):
        skills = [s for s in skills if s.lower() != skill.lower()]
        must_have_skills = [s for s in must_have_skills if s.lower() != skill.lower()]

    current_requirements["skills"] = skills
    current_requirements["must_have_skills"] = must_have_skills

    new_experience = changes.get("min_years_experience")

    if new_experience is not None:
        current_requirements["min_years_experience"] = new_experience

    if changes.get("remove_experience_requirement"):
        current_requirements["min_years_experience"] = 0

    return {"job_requirements": current_requirements}


def compare_candidates(state: AgentState) -> dict:
    candidates = state.get("shortlist", [])

    details = state["intent_details"]
    count = details.get("candidate_count")

    if not candidates:
        return {
            "report": "No candidate results are available. Please run a search first"
        }

    selected = candidates[:count]

    report = []
    report.append(f"COMPARISON OF TOP {len(selected)} CANDIDATES")
    report.append("=" * 50)

    for index, candidate in enumerate(selected, start=1):
        report.append(f"\n{index}. {candidate['candidate_name']}")
        report.append(f"Match score: {candidate['match_score']:.2f}/100")
        report.append(f"Eligibility: {candidate['eligibility']}")
        report.append(
            f"Matched skills: {', '.join(candidate.get('matched_skills', []))}"
        )
        report.append(
            f"Missing skills: {', '.join(candidate.get('missing_skills', []))}"
        )
        report.append(f"Explanation: {candidate.get('explanation', 'Not available')}")

    return {"report": "\n".join(report)}


def explain_ranking(state: AgentState) -> dict:
    candidates = state.get("shortlist", [])
    question = state.get("follow_up_query", "")

    if not candidates:
        return {
            "report": "No candidate results are available. Please run a search first"
        }

    candidate_data = []

    for candidate in candidates:
        candidate_data.append(
            {
                "name": candidate["candidate_name"],
                "match_score": candidate["match_score"],
                "eligibility": candidate["eligibility"],
                "matched_skills": candidate.get("matched_skills", []),
                "missing_skills": candidate.get("missing_skills", []),
                "explanation": candidate.get("explanation", ""),
            }
        )

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a recruitment analysis assistant. "
                    "Explain candidate rankings using only the supplied data. "
                    "Compare scores, matched skills, missing skills, and eligibility. "
                    "Never invent experience or qualifications. "
                    "If evidence is insufficient, clearly say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Candidate data:\n{json.dumps(candidate_data, indent=2)}"
                ),
            },
        ],
        temperature=0,
    )

    explanation = response.choices[0].message.content

    return {"report": explanation}


def generate_questions(state: AgentState) -> dict:
    candidates = state.get("shortlist", [])
    details = state.get("intent_details", {})

    if not candidates:
        return {"report": "No candidates are available. Please run a search first."}

    candidate_names = details.get("candidate_names", [])

    if candidate_names:
        candidate = next(
            (
                c
                for c in candidates
                if c["candidate_name"].lower()
                in [name.lower() for name in candidate_names]
            ),
            None,
        )
    else:
        candidate = candidates[0]

    if candidate is None:
        return {"report": "The requested candidate was not found in the shortlist."}

    candidate_data = {
        "name": candidate["candidate_name"],
        "matched_skills": candidate.get("matched_skills", []),
        "missing_skills": candidate.get("missing_skills", []),
        "explanation": candidate.get("explanation", ""),
    }

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced technical interviewer. "
                    "Generate 8 relevant interview questions for the candidate. "
                    "Include technical questions, practical scenario questions, "
                    "and questions that explore potential skill gaps. "
                    "Do not assume qualifications that are not in the supplied data."
                ),
            },
            {"role": "user", "content": json.dumps(candidate_data, indent=2)},
        ],
        temperature=0.4,
    )

    questions = response.choices[0].message.content

    return {
        "report": (
            f"INTERVIEW QUESTIONS FOR {candidate["candidate_name"]}\n"
            f"{'=' * 50}\n\n{questions}"
        )
    }


def general_question(state: AgentState) -> dict:
    question = state.get("follow_up_query", "")

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are helpful recruitment assistant. "
                    "Answer questions about the recruitment process, "
                    "candidate matching, eligibility, and this agent's capabilities. "
                    "Do not invent candidate information. "
                    "If the user asks something outside your knowledge, say so."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )

    return {"report": response.choices[0].message.content}


def deep_screen_candidates(state: AgentState) -> dict:
    requirements = state["job_requirements"]
    shortlist = state["shortlist"]

    deep_screened = []

    for candidate in shortlist:
        prompt = f"""
You are an expert recruitment screening assistant.

Analyze the candidate's resume against the job requirements.


JOB_REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CANDIDATE:
{candidate['candidate_name']}

RESUME:
{candidate.get('resume_text', '')}

FIRST-ROUND MATCH INFORMATION:
{candidate['explanation']}

Provide your analysis in the following sections:

1. Relevant experience
2. Technical skills demonstrated with resume evidence
3. Strengths relevant to the job
4. Missing skills or experience gaps
5. Potential concerns or unclear information
6. Suggested interview topics

Important:
- Use only information supported by resume.
- Do not invent experience, skills, or qualifications.
- If information is missing, say "Not clearly stated in resume".
- Do not make a final hiring decision based only on the match score.
"""

        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful recruitment analyst. Base every conclusion on the supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        analysis = response.choices[0].message.content

        deep_screened.append({**candidate, "deep_analysis": analysis})

    return {"shortlist": deep_screened}


def make_final_recommendations(state: AgentState) -> dict:
    shortlist = state["shortlist"]
    requirements = state["job_requirements"]

    recommendations = []

    for candidate in shortlist:
        prompt = f"""
You are a recruitment decision-support assistant.

Evaluate this candidate against the job requirements.

JOB REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CANDIDATE:
{candidate['candidate_name']}

MATCH SCORE:
{candidate['match_score']}

ELIGIBILITY:
{candidate['eligibility']}

MATCHED SKILLS:
{json.dumps(candidate['matched_skills'])}

MISSING SKILLS:
{json.dumps(candidate['missing_skills'])}

FIRST ROUND EXPLANATION:
{candidate['explanation']}

SECOND ROUND DEEP ANALYSIS:
{candidate.get('deep_analysis', 'Not available')}

Choose exactly one recommendation:
- Recommend for interview
- Further review required
- Do not recommend at this stage

Return your answer in this format:

Recommendation: [one of the three options]

Reasoning:
[Explain the decision using evidence from the resume and screening results.]

Strengths:
[List relevant strengths.]

Concerns:
[List gaps or uncertainties.]

Important rules:
- Do not invent candidate information.
- Do not assume missing information means the candidate lacks a skill.
- A candidate marked Not eligible must not receive Recommend for interview.
- An Unknown experience status requires Further review unless another requirement makes the candidate ineligible.
- This is decision support, not an automatic hiring decision.
"""
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fair, evidence-based recruitment decision-support assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        recommendation_text = response.choices[0].message.content or ""

        # Enfore eligibility constraints even if the model ignores them.
        if candidate["eligibility"] == "Not eligible":
            recommendation = "Do not recommend at this stage"
        elif candidate["eligibility"] == "Review":
            recommendation = "Further review required"
        else:
            valid_options = [
                "Recommend for interview",
                "Further review required",
                "Do not recommend at this stage",
            ]

            recommendation = next(
                (
                    option
                    for option in valid_options
                    if option.lower() in recommendation_text.lower()
                ),
                "Further review required",
            )

        recommendations.append(
            {
                "candidate_name": candidate["candidate_name"],
                "match_score": candidate["match_score"],
                "eligibility": candidate["eligibility"],
                "recommendation": recommendation,
                "reasoning": recommendation_text,
            }
        )

    return {"final_recommendations": recommendations}


def human_feedback_node(state: AgentState) -> dict:
    recommendations = state.get("final_recommendations", [])

    feedback = interrupt(
        {
            "message": "Please review the recommendations.",
            "instructions": (
                "Enter your feedback, corrections, or approval. "
                "You can also request reassessment."
            ),
            "recommendations": recommendations,
        }
    )

    return {"human_feedback": str(feedback)}


builder = StateGraph(AgentState)


# Main workflow nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("parse_job", parse_job)
builder.add_node("extract_requirements", extract_requirments)
builder.add_node("search_resumes", search_resumes)
builder.add_node("rank_candidates", rank_candidates)
builder.add_node("deep_screen_candidates", deep_screen_candidates)
builder.add_node("make_final_recommendations", make_final_recommendations)
builder.add_node("human_feedback", human_feedback_node)
builder.add_node("generate_report", generate_report)


# Conversational nodes
builder.add_node("refine_requirements", refine_requirements)
builder.add_node("compare_candidates", compare_candidates)
builder.add_node("explain_ranking", explain_ranking)
builder.add_node("generate_questions", generate_questions)
builder.add_node("general_question", general_question)

# Starting point
builder.add_edge(START, "classify_intent")

# Route according to user intent
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "parse_job": "parse_job",
        "refine_requirements": "refine_requirements",
        "compare_candidates": "compare_candidates",
        "explain_ranking": "explain_ranking",
        "generate_questions": "generate_questions",
        "general_question": "general_question",
    },
)

# New job search workflow
builder.add_edge("parse_job", "extract_requirements")
builder.add_edge("extract_requirements", "search_resumes")

# Round 1: Search and rank candidates
builder.add_edge("search_resumes", "rank_candidates")

# Round 2: Deep screening
builder.add_edge("rank_candidates", "deep_screen_candidates")
builder.add_edge("deep_screen_candidates", "make_final_recommendations")
builder.add_edge("make_final_recommendations", "human_feedback")
builder.add_edge("human_feedback", "generate_report")

# Refinement workflow
builder.add_edge("refine_requirements", "search_resumes")


builder.add_edge("generate_report", END)
builder.add_edge("compare_candidates", END)
builder.add_edge("explain_ranking", END)
builder.add_edge("generate_questions", END)

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    messages = []

    # Maintain agent state across conversation turns
    current_state = {
        "messages": [],
        "job_description": "",
        "job_requirements": {},
        "candidates": [],
        "shortlist": [],
        "reasoning": {},
        "report": "",
        "intent": "",
        "intent_details": {},
        "follow_up_query": "",
        "final_recommendations": [],
        "human_feedback": "",
    }

    config = {"configurable": {"thread_id": "resume-matching-test-1"}}

    print("Resume Matching Agent")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        messages.append({"role": "user", "content": query})

        current_state["messages"] = messages

        try:
            result = agent.invoke(current_state, config=config)

            while result.get("__interrupt__"):
                print("\n" + "=" * 45)
                print("RECRUITER REVIEW REQUIRED")
                print("=" * 45)

                for interruption in result["__interrupt__"]:
                    review_data = interruption.value

                    print("\n" + review_data.get("message", "Recruiter review"))
                    print(review_data.get("instructions", ""))

                    print("\n" + "=" * 60)
                    print("CANDIDATE RECOMMENDATIONS")
                    print("=" * 60)

                    for candidate in review_data.get("recommendations", []):
                        print(f"\nCandidate: {candidate['candidate_name']}")
                        print(f"Match Score: {candidate['match_score']}%")
                        print(f"Eligibility: {candidate['eligibility']}")
                        print(f"Recommendation: {candidate['recommendation']}")
                        print("\nReasoning:")
                        print(candidate.get("reasoning", "No reasoning available."))
                        print("-" * 60)

                feedback = input("\nRecruiter feedback / approval: ").strip()

                if not feedback:
                    feedback = "Review completed. No additional feedback."

                # Resume the paused graph
                result = agent.invoke(Command(resume=feedback), config=config)

            # Update application state
            result.pop("__interrupt__", None)
            current_state.update(result)

            report = result.get("report", "")

            messages.append({"role": "assistant", "content": report})

            print("\nAgent: ")
            print(result["report"])
            print()

        except Exception as exc:
            print(f"Error: {exc}")
            traceback.print_exc()
