import os
import re
import json
from pathlib import Path
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "openai/text-embedding-3-small"
DATABASE_FOLDER = "./chroma_db"
COLLECTION_NAME = "resume_chunks"

MUST_HAVE_SKILLS = []

MIN_YEARS_EXPERIENCE = 0

OUTPUT_FILE = "job_match_results.json"

SKILL_TERMS = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data analysis",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "natural language processing",
    "nlp",
    "computer vision",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "html",
    "css",
    "react",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "rest api",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "excel",
    "power bi",
    "tableau",
    "statistics",
    "postgresql",
    "mysql",
    "mongodb",
    "spark",
    "hadoop",
    "etl",
    "data engineering",
    "llm",
    "generative ai",
    "rag",
    "langchain",
]

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

db = chromadb.PersistentClient(path=DATABASE_FOLDER)
collection = db.get_collection(name=COLLECTION_NAME)


def select_job_description():
    job_folder = Path("./job_descriptions")

    job_files = sorted(job_folder.glob("*.txt"), key=lambda path: path.name)

    if not job_files:
        raise FileNotFoundError("No .txt files found in the job_description folder.")

    print("\nAvailable Job Descriptions:")
    for index, file_path in enumerate(job_files, start=1):
        print(f"{index}. {file_path.stem}")

    while True:
        choice = input(f"\nSelect a job description (1-{len(job_files)}): ").strip()

        if choice.isdigit():
            choice = int(choice)

            if 1 <= choice <= len(job_files):
                selected_file = job_files[choice - 1]
                job_description = selected_file.read_text(encoding="utf-8")

                print(f"\nSelected: {selected_file.name}")
                return job_description

        print("Invalid choice. Please enter a valid number.")


def contains_skill(text, skill):
    """Check whether a skill appears in text."""
    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_job_skills(job_description):
    return [skill for skill in SKILL_TERMS if contains_skill(job_description, skill)]


def extract_must_have_skills(job_description, job_skills):

    sentences = re.split(r"[\n.!?;]+", job_description)
    reqiurement_words = [
        "must",
        "required",
        "requirement",
        "require",
        "mandatory",
        "essential",
        "minimum",
        "at least",
    ]

    detected = set(MUST_HAVE_SKILLS)

    for sentence in sentences:
        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in reqiurement_words):
            for skill in job_skills:
                if contains_skill(sentence, skill):
                    detected.add(skill)

    return sorted(
        skill
        for skill in detected
        if any(skill.lower() == job_skill.lower() for job_skill in job_skills)
    )


def get_job_description():
    job_file = Path("job_description.txt")

    if job_file.exists():
        description = job_file.read_text(encoding="utf-8").strip()
        if description:
            print(f"Loaded job description from {job_file}")
            return description

    print("\nPaste the job description below.")
    print("When finished, press Enter on any empty line:\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line.strip():
            break

        lines.append(line)

    return "\n".join(lines).strip()


def get_candidate_name(metadata, document):
    """Get a candidate name, avoiding resume section headings."""

    name = metadata.get("candidate_name", "").strip()

    invalid_names = {
        "",
        "unknown",
        "personal info",
        "personal information",
        "resume",
        "curriculum vitae",
        "cv",
    }

    if name.lower() not in invalid_names:
        return name

    for line in document.splitlines()[:15]:
        line = line.strip()

        if not line:
            continue

        if any(
            word in line.lower()
            for word in [
                "personal info",
                "personal information",
                "contact",
                "email",
                "phone",
                "address",
                "linkedin",
                "github",
                "resume",
            ]
        ):
            continue

        if (
            len(line.split()) <= 4
            and len(line) <= 50
            and line.replace(" ", "").replace("-", "").replace("'", "").isalpha()
        ):
            return line

    filename = os.path.basename(metadata.get("resume_path", ""))
    filename = os.path.splitext(filename)[0]

    filename = re.sub(r"[_\-\s]*(resume|cv)$", "", filename, flags=re.IGNORECASE)

    filename = filename.replace("_", " ").replace("-", " ").strip()

    return filename.title() if filename else "Unknown Candidate"


def build_candidate_profile(results):
    """
    Combine retrieved chunks by resume_id so each candidate
    appears once, retaining the closest matching chunks.
    """

    candidates = {}

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        resume_id = metadata.get("resume_id")

        if not resume_id:
            continue

        if resume_id not in candidates:
            candidates[resume_id] = {
                "name": get_candidate_name(metadata, document),
                "path": metadata.get("resume_path", ""),
                "distance": float(distance),
                "chunks": [],
                "sections": set(),
            }

        candidate = candidates[resume_id]

        candidate["chunks"].append(
            {
                "text": document,
                "distance": float(distance),
                "section": metadata.get("section", "Unknown"),
            }
        )

        candidate["sections"].add(metadata.get("section", "Unknown"))

        candidate["distance"] = min(candidate["distance"], float(distance))

    for candidate in candidates.values():
        candidate["chunks"].sort(key=lambda chunk: chunk["distance"])
        candidate["all_text"] = "\n".join(
            chunk["text"] for chunk in candidate["chunks"]
        )

    return candidates


def score_candidate(candidate, job_skills):
    """
    Score candidates using semantic similarity and skill coverage.
    The score is a ranking aid, not a probability.
    """
    distance = candidate["distance"]

    semantic_similarity = max(0.0, min(1.0, 1.0 - distance))

    resume_text = candidate["all_text"]

    matched_skills = [
        skill for skill in job_skills if contains_skill(resume_text, skill)
    ]

    missing_skills = [skill for skill in job_skills if skill not in matched_skills]

    if job_skills:
        skill_coverage = len(matched_skills) / len(job_skills)
        score = 100 * (0.45 * semantic_similarity + 0.55 * skill_coverage)
    else:
        skill_coverage = None
        score = 100 * semantic_similarity

    return {
        "match_score": round(score, 2),
        "semantic_similarity": round(semantic_similarity, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_coverage": (
            round(skill_coverage, 4) if skill_coverage is not None else None
        ),
    }


def make_explanation(candidate, matched_skills, missing_skills):
    """Create a short explanation grounded in resume text."""
    if matched_skills:
        matched_text = ", ".join(matched_skills)
    else:
        matched_text = "No listed job skills found in retrieved text"

    if missing_skills:
        missing_text = ", ".join(missing_skills)
    else:
        missing_text = "None of the listed job skills are missing"

    return (
        f"Matching evidence: {matched_text}."
        f"Skills not found in retrieved resume text: {missing_text}."
        "Check the original resume before making a decision."
    )


def match_candidates():
    job_desciption = select_job_description()

    if not job_desciption:
        print("No job description provided. Exiting.")
        return

    if collection.count() == 0:
        print("The resume database is empty. Run resume_rag.py first.")
        return

    print("\nExtracting job skills...")

    job_skills = extract_job_skills(job_desciption)
    must_have_skills = extract_must_have_skills(job_desciption, job_skills)

    print("Job skills: ", job_skills or "None detected")
    print("Must have skills: ", must_have_skills or "None detected")

    print("\nGenerating job description embedding...")

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[job_desciption])

    job_embedding = response.data[0].embedding

    print("Searching the resume database...")

    result_count = min(200, collection.count())

    results = collection.query(
        query_embeddings=[job_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    candidates = build_candidate_profile(results)

    if not candidates:
        print("No candidate profiles were retrieved.")
        return

    ranked = []

    for resume_id, candidate in candidates.items():
        score_data = score_candidate(candidate, job_skills)

        missing_must_haves = [
            skill
            for skill in must_have_skills
            if not contains_skill(candidate["all_text"], skill)
        ]

        eligible = not missing_must_haves

        excerpts = [chunk["text"][:350] for chunk in candidate["chunks"][:3]]

        ranked.append(
            {
                "resume_id": resume_id,
                "candidate_name": candidate["name"],
                "resume_path": candidate["path"],
                "match_score": score_data["match_score"],
                "semantic_similarity": score_data["semantic_similarity"],
                "matched_skills": score_data["matched_skills"],
                "missing_skills": score_data["missing_skills"],
                "must_have_skills": must_have_skills,
                "missing_must_have_skills": missing_must_haves,
                "eligibility": (
                    "Eligible"
                    if eligible
                    else "Review / does not meet configured requirements"
                ),
                "sections_retrieved": sorted(candidate["sections"]),
                "relevant_excerpts": excerpts,
                "explanation": make_explanation(
                    candidate,
                    score_data["matched_skills"],
                    score_data["missing_skills"],
                ),
            }
        )

    ranked.sort(key=lambda item: item["match_score"], reverse=True)

    eligible_candidates = [
        candidate for candidate in ranked if candidate["eligibility"] == "Eligible"
    ]

    if eligible_candidates:
        top_candidates = eligible_candidates[:10]
        selection_note = "Top eligible candidates"
    else:
        top_candidates = ranked[:10]
        selection_note = (
            "No candidates passed all configured requirements. "
            "Showing the highest-scoring candidates for manual review."
        )

    output = {
        "job_description": job_desciption,
        "job_skills": job_skills,
        "must_have_skills": must_have_skills,
        "scoring_method": (
            "45% semantic similarity + 55% job-skill coverage "
            "when job skills are detected; otherwise semantic similarity"
        ),
        "selection_note": selection_note,
        "total_unique_candidates_retrieved": len(ranked),
        "top_matches": top_candidates,
        "excluded_by_requirements": [
            candidate for candidate in ranked if candidate["eligibility"] != "Eligible"
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(selection_note)
    print("=" * 65)

    for rank, candidate in enumerate(top_candidates, start=1):
        print(f"\n{rank}. {candidate['candidate_name']}")
        print(f"   Match score: {candidate['match_score']}/100")
        print(f"   Eligibility: {candidate['eligibility']}")
        print(
            "   Matched skills:", ", ".join(candidate["matched_skills"]) or "None found"
        )

        if candidate["missing_must_have_skills"]:
            print(
                "   Missing must-haves:",
                ", ".join(candidate["missing_must_have_skills"]),
            )

        print(f"   Explanation: {candidate['explanation']}")
        print(f"   Resume: {candidate['resume_path']}")

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "\nNote: Matching only checks the retrieved resume text. "
        "Review original resumes; missing text does not prove a "
        "candidate lacks a skill."
    )


if __name__ == "__main__":
    match_candidates()
