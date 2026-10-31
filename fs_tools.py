import os
from pypdf import PdfReader
from docx import Document
from pathlib import Path


def read_file(filepath: str) -> dict:
    try:
        extension = os.path.splitext(filepath)[1].lower()
        if extension == ".txt":
            with open(filepath, "r") as file:
                content = file.read()
        elif extension == ".pdf":
            reader = PdfReader(filepath)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
        elif extension == ".docx":
            document = Document(filepath)
            content = ""
            for paragraph in document.paragraphs:
                content += (paragraph.text or "") + "\n"
        result = {
            "success": True,
            "content": content,
            "metadata": {
                "filename": os.path.basename(filepath),
                "extension": extension,
                "modified_date": os.path.getmtime(filepath),
                "size": os.path.getsize(filepath),
            },
        }
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_metadata(filepath: str) -> dict:
    return {
        "name": os.path.basename(filepath),
        "size": os.path.getsize(filepath),
        "modified_date": os.path.getmtime(filepath),
    }


def list_files(directory: str, extension: str = None) -> list:
    if extension == "":
        extension = None
    files = os.listdir(directory)
    result = []
    if extension is not None:
        extension = extension.lower()
    for f in files:
        filepath = os.path.join(directory, f)

        file_extension = os.path.splitext(filepath)[1].lower()
        if extension is None or file_extension == extension:
            result.append(build_metadata(filepath))
    return result


def write_file(filepath: str, content: str) -> dict:
    try:
        path = Path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as file:
            file.write(content)
        return {"success": True, "filepath": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_in_file(filepath: str, keyword: str) -> dict:
    try:
        result = read_file(filepath)
        if result["success"] == True:
            content = result["content"]
            position = content.lower().find(keyword.lower())
            if position == -1:
                return {"success": True, "context": None}
            else:
                # Start 50 characters before keyword, but never go before the beginning of file
                start = max(0, position - 50)
                # Start at the keyword and include keyword plus up to 50 characters after it
                end = position + len(keyword) + 50
                context = content[start:end]
                return {"success": True, "context": context}
    except Exception as e:
        return {"success": False, "error": str(e)}
