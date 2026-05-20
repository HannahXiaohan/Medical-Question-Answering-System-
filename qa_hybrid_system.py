
import re
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ----------------------------
# MongoDB
# ----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["MedlineHealth"]

enc_col = db["Encyclopedia_KB"]
vec_col = db["Vector_KB"]

# ----------------------------
# Model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Utils
# ----------------------------
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_entity(q):
    remove = ["what", "is", "are", "the", "of",
              "causes", "cause", "symptoms",
              "treatment", "exams", "tests", "prevention"]

    words = [w for w in clean(q).split() if w not in remove]
    return " ".join(words)


# ----------------------------
# Parse Question
# ----------------------------
def parse(q):
    ql = clean(q)

    if "symptom" in ql:
        return "Symptoms", normalize_entity(q)

    if "cause" in ql:
        return "Causes", normalize_entity(q)

    if "treatment" in ql:
        return "Treatment", normalize_entity(q)

    if "exam" in ql or "test" in ql:
        return "Exams", normalize_entity(q)

    if "prevent" in ql:
        return "Prevention", normalize_entity(q)

    return "definition", normalize_entity(q)


# ----------------------------
# Find topic
# ----------------------------
def find_doc(entity):
    return enc_col.find_one({
        "topic_name": {"$regex": f"^{entity}$", "$options": "i"}
    })


# ----------------------------
# Section answers
# ----------------------------
def get_section(entity, q_type):
    doc = find_doc(entity)
    if not doc:
        return []

    answers = []

    for sec in doc.get("sections", []):
        if q_type.lower() not in sec["section_name"].lower():
            continue

        for lst in sec.get("lists", []):
            for item in lst.get("list_items", []):
                if item:
                    answers.append(item)

        for sent in sec.get("sentences", []):
            text = sent.get("sentence") or sent.get("text")
            if text:
                answers.append(text)

    return list(dict.fromkeys(answers))


# ----------------------------
# Summary
# ----------------------------
def get_summary(entity):
    doc = find_doc(entity)
    if not doc:
        return []

    return [s["sentence"] for s in doc.get("summary", [])]


# ----------------------------
# Vector fallback
# ----------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_search(q):
    q_vec = model.encode(q)

    docs = list(vec_col.find().limit(800))
    scored = []

    for d in docs:
        vec = d.get("vector") or d.get("embedding")
        if not vec:
            continue

        score = cosine(q_vec, vec)
        scored.append((score, d["sentence"]))

    scored.sort(reverse=True)
    return [s for _, s in scored[:3]]


# ----------------------------
# Clean sentence
# ----------------------------
def refine(text, entity):
    text = text.strip()

    text = re.sub(rf"\b{entity}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"is (usually )?caused by", "", text, flags=re.IGNORECASE)
    text = re.sub(r"the bacterium called", "the bacterium", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.+$", ".", text)

    return text.strip()


# ----------------------------
# Generate Answer
# ----------------------------
def generate(entity, q_type, answers):
    if not answers:
        return None

    entity_cap = entity.capitalize()

    if q_type == "definition":
        return answers[0]

    lines = []
    for a in answers:
        clean_a = refine(a, entity)
        if clean_a:
            lines.append(f"- {clean_a}")

    title_map = {
        "Causes": "causes include",
        "Treatment": "treatments include",
        "Symptoms": "symptoms include",
        "Prevention": "prevention includes",
        "Exams": "diagnosis methods include"
    }

    title = title_map.get(q_type, "includes")

    return f"{entity_cap} {title}:\n" + "\n".join(lines)


# ----------------------------
# QA Pipeline
# ----------------------------
def answer(q):
    q_type, entity = parse(q)

    print("\nQuestion:", q)
    print("Type:", q_type)
    print("Entity:", entity)

    # 1. summary
    if q_type == "definition":
        ans = get_summary(entity)
        final = generate(entity, q_type, ans)

        if final:
            print("\nAnswer:")
            print(final)
            return

    # 2. section
    ans = get_section(entity, q_type)
    if ans:
        final = generate(entity, q_type, ans)

        print("\nAnswer:")
        print(final)
        return

    # 3. vector fallback
    ans = vector_search(q)
    if ans:
        print("\nAnswer:")
        print(ans[0])
        return

    print("\nNo answer found.")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    print("🎯 CLEAN QA SYSTEM")

    while True:
        q = input("\nAsk: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer(q)
        print("\n" + "="*60)