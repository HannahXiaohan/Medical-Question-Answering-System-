import spacy
import requests
from pymongo import MongoClient
import time

# Load NLP model
nlp = spacy.load("en_core_sci_sm")

# Stanford OpenIE server URL
OPENIE_URL = "http://localhost:9000/?properties={\"annotators\":\"openie\",\"outputFormat\":\"json\"}"

# Important Sections
IMPORTANT_SECTIONS = [
    "Symptoms", "Causes", "Exams", "Tests",
    "Treatment", "Risk", "Complications"
]

def is_important(section_name):
    return any(k.lower() in section_name.lower() for k in IMPORTANT_SECTIONS)

# Entity Filtering
STOP_ENTITIES = {
    "patient", "people", "person",
    "symptoms", "symptom",
    "condition", "conditions",
    "cases", "case",
    "provider", "providers",
    "health", "health care",
    "body", "size", "risk",
    "children", "child",
    "blood", "skin",
    "disease", "severe",
    "medicine", "medicines",
    "surgery", "problem", "issue", "type"
}

MEDICAL_HINTS = [
    "cancer", "diabetes", "infection",
    "syndrome", "tumor",
    "hormone", "virus", "bacteria",
    "fever", "pain", "injury",
    "arthritis", "asthma"
]

def is_medical_entity(e):
    return any(k in e for k in MEDICAL_HINTS)

def extract_entities(text):
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        e = ent.text.lower().strip()

        if len(e) <= 2:
            continue

        if e in STOP_ENTITIES:
            continue

        # keep multi-word OR medical terms
        if not is_medical_entity(e) and len(e.split()) == 1:
            continue

        entities.add(e)

    return list(entities)


# Negation detection
def detect_negation(text):
    neg_words = ["no", "not", "never", "without", "none"]
    return any(word in text.lower() for word in neg_words)


# Clean triples
def clean_triples(triples):
    cleaned = []

    for t in triples:
        subj = t.get("subject", "").strip().lower()
        rel = t.get("relation", "").strip().lower()
        obj = t.get("object", "").strip().lower()


        if len(subj) < 3 or len(obj) < 3:
            continue

        if subj in ["it", "this", "that"]:
            continue

        if obj in ["something", "anything", "examples", "things", "types"]:
            continue

        if subj == obj:
            continue

        if rel in ["is", "are"] and obj in ["disease", "diseases"]:
            continue

        obj = obj.split()[-1]

        cleaned.append({
            "subject": subj,
            "relation": rel,
            "object": obj
        })

    return cleaned


# Extract OpenIE triples
def extract_triples(text):
    triples = []

    try:
        res = requests.post(OPENIE_URL, data=text.encode("utf-8"), timeout=3)
        data = res.json()

        for sent in data.get("sentences", []):
            for triple in sent.get("openie", []):
                triples.append({
                    "subject": triple.get("subject", ""),
                    "relation": triple.get("relation", ""),
                    "object": triple.get("object", "")
                })

    except Exception:

        return []

    return clean_triples(triples)


# Process one document
def process_document(doc):

    for sec in doc.get("sections", []):

        if not is_important(sec.get("section_name", "")):
            continue

        for sent in sec.get("sentences", []):

            text = sent.get("sentence", "")

            if not text:
                continue

            sent["entities"] = extract_entities(text)
            sent["triples"] = extract_triples(text)
            sent["negation"] = detect_negation(text)

    return doc

# MAIN
def main():

    client = MongoClient("mongodb://localhost:27017/")
    db = client["MedlineHealth"]

    source_col = db["Encyclopedia"]
    target_col = db["Encyclopedia_KB"]

    target_col.delete_many({})

    docs = list(source_col.find())

    print("Total documents:", len(docs))

    for i, doc in enumerate(docs):
        print(f"[{i+1}/{len(docs)}] Processing: {doc.get('topic_name')}")

        new_doc = process_document(doc)
        target_col.insert_one(new_doc)

        time.sleep(0.02)  # 更快

    print("\nDone. Data saved to Encyclopedia_KB")


# RUN
if __name__ == "__main__":
    main()
