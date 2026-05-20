from pymongo import MongoClient

# -----------------------------
# MongoDB setup
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["MedlineHealth"]

source_col = db["Encyclopedia_KB"]
target_col = db["OIE_KB"]

target_col.delete_many({})


# -----------------------------
# Build OIE KB
# -----------------------------
def build_oie_kb():

    docs = list(source_col.find())
    print("Total documents:", len(docs))

    records = []

    tuple_seen = set()
    triple_seen = set()

    total_records = 0

    # -----------------------------
    # Loop documents
    # -----------------------------
    for i, doc in enumerate(docs):
        topic = doc.get("topic_name", "")

        print(f"[{i+1}/{len(docs)}] Processing: {topic}")

        for sec in doc.get("sections", []):
            section_name = sec.get("section_name", "")

            # -------------------------
            # 1. Tuple (topic + section)
            # -------------------------
            tuple_key = (topic, section_name)

            if tuple_key not in tuple_seen:
                tuple_seen.add(tuple_key)

                records.append({
                    "type": "tuple",
                    "topic": topic,
                    "section": section_name
                })
                total_records += 1

            # -------------------------
            # 2. Triples
            # -------------------------
            for sent in sec.get("sentences", []):

                for t in sent.get("triples", []):

                    subj = t.get("subject", "").strip().lower()
                    rel = t.get("relation", "").strip().lower()
                    obj = t.get("object", "").strip().lower()

                    # --- basic filter ---
                    if not subj or not obj:
                        continue

                    if len(subj) < 3 or len(obj) < 3:
                        continue

                    # --- remove noisy triples ---
                    if subj in ["it", "this", "that"]:
                        continue

                    if obj in ["examples", "things", "something", "anything"]:
                        continue

                    # --- deduplicate ---
                    triple_key = (topic, section_name, subj, rel, obj)

                    if triple_key in triple_seen:
                        continue

                    triple_seen.add(triple_key)

                    # --- definition flag ---
                    is_definition = rel in ["is", "are", "refers to"]

                    records.append({
                        "type": "triple",
                        "topic": topic,
                        "section": section_name,
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "negation": sent.get("negation", False),
                        "is_definition": is_definition
                    })

                    total_records += 1

            # -------------------------
            # 3. Quads (list items)
            # -------------------------
            for lst in sec.get("lists", []):
                list_name = lst.get("list_name", "").strip()

                for item in lst.get("list_items", []):

                    if not item:
                        continue

                    records.append({
                        "type": "quad",
                        "topic": topic,
                        "section": section_name,
                        "relation": list_name,
                        "object": item.strip()
                    })

                    total_records += 1

    # -----------------------------
    # Bulk insert
    # -----------------------------
    if records:
        target_col.insert_many(records)

    print("Total records inserted:", total_records)

    # -----------------------------
    # Indexes
    # -----------------------------
    print("Creating indexes...")

    target_col.create_index("subject")
    target_col.create_index("object")
    target_col.create_index("relation")
    target_col.create_index("topic")
    target_col.create_index("section")
    target_col.create_index("type")
    target_col.create_index("is_definition")

    print("Indexes created.")
    print("Done.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    build_oie_kb()