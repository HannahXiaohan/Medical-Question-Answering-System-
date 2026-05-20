import json
import re
import requests
import time
from bs4 import BeautifulSoup
from pymongo import MongoClient
from web_scraping import get_topic_links


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)


def extract_sections(soup):
    sections = []

    section_divs = soup.find_all("div", class_="section")

    for sid, sec in enumerate(section_divs):

        h2 = sec.find("h2")
        if not h2:
            continue

        section_name = h2.get_text(strip=True)

        body = sec.find("div", class_="section-body")
        if not body:
            continue

        sentences = []
        lists = []

        # ---------- sentences ----------
        sentence_id = 0
        for p in body.find_all("p"):
            text = p.get_text(" ", strip=True)

            # ❌ 跳过 list 引导句
            if text.endswith(":"):
                continue

            for s in split_sentences(text):
                if s:
                    sentences.append({
                        "sentence_id": sentence_id,
                        "sentence": s
                    })
                    sentence_id += 1

        # ---------- lists ----------
        list_id = 0
        for ul in body.find_all("ul"):
            items = []

            for li in ul.find_all("li"):
                item_text = li.get_text(" ", strip=True)
                if item_text:
                    items.append(item_text)

            if items:
                prev_p = ul.find_previous("p")
                list_name = prev_p.get_text(strip=True) if prev_p else section_name

                lists.append({
                    "list_id": list_id,
                    "list_name": list_name,
                    "list_items": items
                })
                list_id += 1

        sections.append({
            "section_id": sid,
            "section_name": section_name,
            "sentences": sentences,
            "lists": lists
        })

    return sections

def parse_article(topic_id, item):
    url = item["url"]
    title = item["title"]

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)

    soup = BeautifulSoup(response.text, "html.parser")

    # ---------- summary ----------
    summary = []
    summary_container = soup.find("div", id="ency_summary")

    if summary_container:
        for p in summary_container.find_all("p"):
            text = p.get_text(" ", strip=True)

            for s in split_sentences(text):
                if s:
                    summary.append({
                        "sentence_id": len(summary),
                        "sentence": s
                    })

    # ---------- sections ----------
    sections = extract_sections(soup)

    return {
        "topic_id": topic_id,
        "topic_name": title,
        "url": url,
        "summary": summary,
        "sections": sections
    }


def build_dataset(topic_links):
    dataset = []

    total = len(topic_links)

    for i, item in enumerate(topic_links):
        print(f"[{i+1}/{total}] Processing: {item['title']}")

        try:
            data = parse_article(i, item)
            dataset.append(data)

            # 🔥 防止被封
            time.sleep(0.2)

        except Exception as e:
            print(f"Failed: {item['url']} | {e}")

    return dataset



def save_json(data, filename="medline_dataset.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_mongodb(data):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MedlineHealth"]
    col = db["Encyclopedia"]

    col.delete_many({})
    if data:
        col.insert_many(data)

    print(f"Inserted {len(data)} documents into MongoDB")



if __name__ == "__main__":
    topic_links = get_topic_links()
    print("Total links:", len(topic_links))  # 应该 ≈ 4438

    dataset = build_dataset(topic_links)

    save_json(dataset)
    save_mongodb(dataset)