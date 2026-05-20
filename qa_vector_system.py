from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# =============================
# MongoDB
# =============================
client = MongoClient("mongodb://localhost:27017/")
db = client["MedlineHealth"]

vector_col = db["Vector_KB"]

# =============================
# Embedding Model
# =============================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================
# Utility
# =============================
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "with", "about", "explain", "why", "how", "does", "do",
    "tell", "me", "please", "can", "you", "give", "list", "show"
}


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================
# Question Type Detection
# =============================
def detect_question_type(question):
    q = question.lower()

    if any(w in q for w in ["symptom", "symptoms", "sign", "signs"]):
        return "Symptoms"

    if any(w in q for w in ["cause", "causes", "why", "occur", "occurs", "reason"]):
        return "Causes"

    if any(w in q for w in ["treat", "treatment", "therapy", "medicine", "medication"]):
        return "Treatment"

    if any(w in q for w in ["diagnose", "diagnosis", "test", "tests"]):
        return "Diagnosis"

    if any(w in q for w in ["prevent", "prevention"]):
        return "Prevention"

    return "General"


# =============================
# Entity Extraction
# =============================
def extract_entity(question):
    q = question.lower()

    remove_words = {
        "what", "is", "are", "the", "of", "a", "an",
        "symptoms", "symptom", "signs", "sign",
        "causes", "cause", "why", "explain",
        "treatment", "treat", "therapy",
        "diagnosis", "diagnose", "test", "tests",
        "prevention", "prevent",
        "occurs", "occur", "does", "do", "for", "about"
    }

    words = re.sub(r"[^a-z0-9\s]", " ", q).split()
    entity_words = [w for w in words if w not in remove_words]

    if not entity_words:
        return ""

    return " ".join(entity_words).strip()


# =============================
# Section Boost
# =============================
def section_boost(question_type, section):
    section = section.lower()

    boost_map = {
        "Symptoms": ["symptom", "symptoms", "signs"],
        "Causes": ["cause", "causes", "risk", "risk factors"],
        "Treatment": ["treatment", "treat", "therapy", "medicines", "medications"],
        "Diagnosis": ["diagnosis", "diagnose", "tests", "exams"],
        "Prevention": ["prevention", "prevent"]
    }

    if question_type not in boost_map:
        return 0.0

    for key in boost_map[question_type]:
        if key in section:
            return 0.25

    return 0.0


# =============================
# Keyword Overlap Boost
# =============================
def keyword_overlap_score(question, sentence):
    q_words = set(tokenize(question))
    s_words = set(tokenize(sentence))

    if not q_words or not s_words:
        return 0.0

    overlap = q_words.intersection(s_words)
    return len(overlap) / len(q_words)


# =============================
# Improved Vector Search
# =============================
def improved_vector_search(question, top_k=8):
    q_type = detect_question_type(question)
    entity = extract_entity(question)

    query_vec = model.encode(question).tolist()
    q_keywords = tokenize(question)

    scored_results = []

    for doc in vector_col.find():
        sentence = clean_text(doc.get("sentence", ""))
        topic = clean_text(doc.get("topic", ""))
        section = clean_text(doc.get("section", ""))

        if not sentence:
            continue

        vector = doc.get("vector")
        if not vector:
            continue

        vector_score = cosine_similarity(query_vec, vector)

        # Entity/topic boost
        entity_boost = 0.0
        if entity:
            if entity.lower() in topic.lower():
                entity_boost += 0.35
            if entity.lower() in sentence.lower():
                entity_boost += 0.15

        # Section boost
        sec_boost = section_boost(q_type, section)

        # Keyword boost
        keyword_boost = keyword_overlap_score(question, sentence) * 0.20

        # Penalize unrelated topic if entity exists and does not appear anywhere
        penalty = 0.0
        if entity:
            combined = f"{topic} {section} {sentence}".lower()
            entity_terms = entity.lower().split()

            if not any(term in combined for term in entity_terms):
                penalty -= 0.25

        final_score = (
            vector_score * 0.60
            + entity_boost
            + sec_boost
            + keyword_boost
            + penalty
        )

        doc["vector_score"] = vector_score
        doc["final_score"] = final_score
        doc["question_type"] = q_type
        doc["entity"] = entity

        scored_results.append(doc)

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    return scored_results[:top_k], q_type, entity


# =============================
# Format Answer
# =============================
def format_answer(results, q_type, entity):
    if not results:
        return "No answer found."

    sentences = []
    seen = set()

    for r in results:
        sent = clean_text(r.get("sentence", ""))

        if not sent:
            continue

        # Avoid very short useless sentences
        if len(sent.split()) < 4:
            continue

        # Remove duplicates
        normalized = sent.lower()
        if normalized in seen:
            continue

        seen.add(normalized)
        sentences.append(sent)

        if len(sentences) >= 5:
            break

    if not sentences:
        return "No answer found."

    if q_type == "Symptoms":
        header = f"{entity.capitalize()} symptoms include:"
    elif q_type == "Causes":
        header = f"{entity.capitalize()} may occur because:"
    elif q_type == "Treatment":
        header = f"{entity.capitalize()} treatment may include:"
    elif q_type == "Diagnosis":
        header = f"{entity.capitalize()} diagnosis may include:"
    elif q_type == "Prevention":
        header = f"{entity.capitalize()} prevention may include:"
    else:
        header = f"Information about {entity}:"

    answer = header + "\n"
    for s in sentences:
        answer += f"- {s}\n"

    return answer.strip()


# =============================
# Main QA
# =============================
def ask(question):
    results, q_type, entity = improved_vector_search(question)

    answer = format_answer(results, q_type, entity)

    print("\nQuestion:", question)
    print("Type:", q_type)
    print("Entity:", entity)
    print("\nAnswer:")
    print(answer)


# =============================
# Run
# =============================
if __name__ == "__main__":
    print("Medical QA System - Improved Vector Ranking")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        ask(question)