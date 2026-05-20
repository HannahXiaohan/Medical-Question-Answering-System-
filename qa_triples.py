import re
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# MongoDB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["MedlineHealth"]

enc = db["Encyclopedia_KB"]
vec = db["Vector_KB"]

# -----------------------------
# Embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# ENTITY EXTRACTION
# -----------------------------
def extract_entity(question):
    q = question.lower()

    best_match = None
    best_score = 0

    for doc in enc.find({}, {"topic_name": 1}):
        name = doc["topic_name"]
        name_lower = name.lower()

        # exact match
        if name_lower in q:
            return name

        # partial match
        score = sum(1 for w in q.split() if w in name_lower)

        if score > best_score:
            best_score = score
            best_match = name

    return best_match


# -----------------------------
# QUESTION TYPE
# -----------------------------
def detect_type(question):
    q = question.lower()

    if "cause" in q or "why" in q:
        return "Causes"
    if "symptom" in q:
        return "Symptoms"
    if "prevent" in q:
        return "Prevention"
    if "treatment" in q:
        return "Treatment"

    return "definition"


# -----------------------------
# FIND TOPIC
# -----------------------------
def find_topic(entity):
    return enc.find_one({
        "topic_name": {"$regex": f"^{entity}$", "$options": "i"}
    })


# -----------------------------
# TRIPLE ANSWER
# -----------------------------
def answer_from_triples(entity, q_type):
    doc = find_topic(entity)
    if not doc:
        return None, None

    answers = []
    entity_lower = entity.lower()

    for sec in doc.get("sections", []):
        for sent in sec.get("sentences", []):
            triples = sent.get("triples", [])
            original_sentence = sent.get("sentence", "")

            for t in triples:
                subj = t.get("subject", "").lower()
                rel = t.get("relation", "").lower()
                obj = t.get("object", "").lower()

                if not subj or not obj:
                    continue

                # -------------------------
                # Definition
                # -------------------------
                if q_type == "definition":
                    if entity_lower in subj and rel in ["is", "are", "refers to", "involves"]:
                        if original_sentence:
                            answers.append(original_sentence)
                        else:
                            answers.append(f"{entity.capitalize()} {rel} {obj}.")

                # -------------------------
                # question types
                # -------------------------
                else:
                    if entity_lower in subj:
                        if original_sentence:
                            answers.append(original_sentence)
                        else:
                            answers.append(f"{subj.capitalize()} {rel} {obj}.")

    if answers:
        return list(dict.fromkeys(answers))[:5], "triple"

    return None, None


# -----------------------------
# SECTION ANSWER
# -----------------------------
def answer_from_section(entity, q_type):
    doc = find_topic(entity)
    if not doc:
        return None, None

    for sec in doc.get("sections", []):
        if q_type.lower() in sec.get("section_name", "").lower():
            sentences = [s["sentence"] for s in sec.get("sentences", [])]

            for lst in sec.get("lists", []):
                sentences.extend(lst.get("list_items", []))

            return sentences, "section"

    return None, None


# -----------------------------
# SUMMARY ANSWER
# -----------------------------
def answer_from_summary(entity):
    doc = find_topic(entity)
    if not doc:
        return None, None

    summary = doc.get("summary", [])

    sentences = []
    for s in summary:
        if isinstance(s, dict):
            sentences.append(s.get("sentence", ""))
        else:
            sentences.append(str(s))

    if sentences:
        return sentences, "summary"

    return None, None


# -----------------------------
# VECTOR FALLBACK
# -----------------------------
def answer_from_vector(question):
    docs = list(vec.find().limit(500))

    if not docs:
        return ["No vector data available."]

    query_emb = model.encode(question, convert_to_tensor=True)

    best_score = -1
    best_text = ""

    for d in docs:
        emb = d.get("vector")
        if not emb:
            continue

        score = util.cos_sim(query_emb, emb)[0][0].item()

        if score > best_score:
            best_score = score
            best_text = d.get("sentence", "")

    return [best_text]


# -----------------------------
# FORMAT OUTPUT
# -----------------------------
def format_answer(entity, answers):
    result = f"\n{entity}:\n"

    for a in answers:
        result += f"• {a.strip()}\n"

    return result


# -----------------------------
#  MAIN QA FUNCTION
# -----------------------------
def qa_system(question):
    output = []
    output.append("======================================")
    output.append(f"Question: {question}")

    entity = extract_entity(question)
    q_type = detect_type(question)

    output.append(f"Type: {q_type}")
    output.append(f"Entity: {entity}")

    if not entity:
        return "\n".join(output + ["No entity detected."])

    # -------------------------
    # 1. TRIPLE FIRST
    # -------------------------
    answer, source = answer_from_triples(entity, q_type)
    if answer:
        output.append("\nAnswer (from triples):")
        output.append(format_answer(entity, answer))
        return "\n".join(output)

    # -------------------------
    # 2. SECTION
    # -------------------------
    answer, source = answer_from_section(entity, q_type)
    if answer:
        output.append("\nAnswer (from section):")
        output.append(format_answer(entity, answer))
        return "\n".join(output)

    # -------------------------
    # 3. SUMMARY
    # -------------------------
    answer, source = answer_from_summary(entity)
    if answer:
        output.append("\nAnswer (from summary):")
        output.append(format_answer(entity, answer))
        return "\n".join(output)

    # -------------------------
    # 4. VECTOR FALLBACK
    # -------------------------
    answer = answer_from_vector(question)
    output.append("\nAnswer (from vector):")
    output.append(format_answer(entity, answer))

    return "\n".join(output)


# -----------------------------
# RUN LOOP
# -----------------------------
if __name__ == "__main__":
    print("🔥 Triple-Based QA System (Final)")
    print("Type 'exit' to quit.")

    while True:
        q = input("\nAsk: ")

        if q.lower() in ["exit", "quit"]:
            break

        if not q.strip():
            continue

        result = qa_system(q)
        print(result)