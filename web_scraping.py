import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_topic_links():
    base = "https://medlineplus.gov/ency/encyclopedia_{}.htm"
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["0-9"]

    all_links = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for letter in letters:
        url = base.format(letter)
        print(f"Scraping index: {url}")

        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.select("#index a"):
                title = a.get_text(strip=True)
                href = a.get("href")

                if not href:
                    continue

                full_url = urljoin(url, href)

                if "/ency/article/" in full_url or "/ency/patientinstructions/" in full_url:
                    all_links.append({
                        "title": title,
                        "url": full_url
                    })

        except Exception as e:
            print(f"Failed: {url} | {e}")

    print("Total links:", len(all_links))
    return all_links


if __name__ == "__main__":
    topic_links = get_topic_links()
    print("Total links:", len(topic_links))

    for i in range(min(10, len(topic_links))):
        print(i + 1, topic_links[i])