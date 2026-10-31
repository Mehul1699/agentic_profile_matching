print("STEP 1: Script started", flush=True)

print("Before standard library imports", flush=True)
import os
import re
import hashlib
from pathlib import Path

print("Before ChromaDB import", flush=True)
import chromadb

print("Before dotenv import", flush=True)
from dotenv import load_dotenv

print("Before OpenAI import", flush=True)
from openai import OpenAI

print("Before file_utils import", flush=True)
from fs_tools import read_file

print("All imports completed", flush=True)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"
RESUME_FOLDER = Path("resumes")
DATABASE_FOLDER = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=DATABASE_FOLDER)

collection = chroma_client.get_or_create_collection(
    name="resume_chunks",
    metadata={"hnsw:space" : "cosine"}
)

print("STEP 2: ChromaDB collection created", flush=True)

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def chunk_resume(text: str) -> list[dict]:
    section_names = {
        "summary": "Summary",
        "professional summary": "Summary",
        "skills": "Skills",
        "technical skills": "Skills",
        "experience": "Experience",
        "work experience": "Experience",
        "professional experience": "Experience",
        "education": "Education",
        "projects": "Projects",
        "certifications": "Certifications",
    }

    chunks = []
    current_section = "General"
    current_lines = []

    def save_chunk():
        content = "\n".join(current_lines).strip()

        if content:
            words = content.split()
            chunk_size = 180
            overlap = 30

            start = 0
            while start < len(words):
                part = " ".join(words[start:start + chunk_size])

                chunks.append({
                    "section" : current_section,
                    "text" : f"Section: {current_section}\n{part}"
                })

                if start + chunk_size >= len(words):
                    break

                start += chunk_size - overlap

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        heading = re.sub(r"[:\s]+$", "", line).lower()

        if heading in section_names:
            save_chunk()
            current_section = section_names[heading]
            current_lines = []
        else:
            current_lines.append(line)

    save_chunk()
    return chunks



def extract_candidate_name(text: str, filepath: Path) -> str:
    """Try to identify the candidate's name near the start of a resume."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Examine the first few lines, where a resume name usually appears.
    for line in lines[:8]:
        cleaned = line.strip(" |:-\t")

        # Skip contact details, headings, and obvious labels.
        if any(char.isdigit() for char in cleaned):
            continue

        if "@" in cleaned or "http" in cleaned.lower():
            continue

        if len(cleaned.split()) not in (2, 3, 4):
            continue

        if len(cleaned) > 60:
            continue

        if cleaned.lower() in {
            "resume",
            "curriculum vitae",
            "professional summary",
            "technical skills",
            "work experience",
            "education",
        }:
            continue

        # Accept typical names made up of alphabetic words,
        # including common hyphenated or apostrophe names.
        words = cleaned.split()

        if all(
            re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word)
            for word in words
        ):
            return cleaned

    # Fall back to the filename if no plausible name is found.
    return filepath.stem


def index_resumes():
    print("STEP 3: Entered index_resumes()", flush=True)
    if not RESUME_FOLDER.exists():
        print(f"Resume folder not found: {RESUME_FOLDER.resolve()}")
        return

    indexed_files = 0

    for filepath in RESUME_FOLDER.iterdir():
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue

        result = read_file(str(filepath))

        if not result.get("success"):
            print(f"Could not read {filepath.name}: {result.get('error')}")
            continue


        text = result.get("content", "").strip()


        if not text:
            print(f"No text extracted from {filepath.name}")
            continue

        chunks = chunk_resume(text)

        if not chunks:
            print(f"No chunks created for {filepath.name}")
            continue

        documents = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(documents)

        file_key = hashlib.sha256(
            str(filepath.resolve()).encode("utf-8")
        ).hexdigest()[:16]

        existing = collection.get(
            where={"resume_id": file_key},
            include=[]
        )


        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        ids = [
            f"{file_key}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "resume_id": file_key,
                "resume_path": str(filepath.resolve()),
                "candidate_name": extract_candidate_name(text, filepath),
                "section": chunk["section"],
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]

        collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas
        )

        indexed_files += 1
        print(f"Finished! Indexed {indexed_files} resume files.")
        print(f"Total chunks in database: {collection.count()}")

    print("STEP 2: Functions defined", flush=True)

if __name__ == "__main__":
    print("STEP 3: Starting resume indexing", flush=True)
    index_resumes()
    print("STEP 4: Resume indexing finished", flush=True)