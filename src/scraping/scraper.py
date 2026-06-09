import csv
import json
import random
import re
import statistics
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

INDEX_PAGE_API = "https://animanch.com/page/{page}"
CONTENT_PAGE_API = "https://animanch.com/archives/{id}.html"
KAKOLOG_PAGE_API = "https://bbs.animanch.com/kakolog/{date}/page:{page}"
BOARD_PAGE_API = "https://bbs.animanch.com/board/{id}/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.5",
}


def get_latest_thread_id():

    ids = []
    for page in range(1, 35):  # page 1 ~ 34
        url = INDEX_PAGE_API.format(page=page)
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        contents = soup.find(id="contents")
        if contents is None:
            print(f"page {page} doesn't have contents panel")
            break
        articles = contents.find_all(class_="entry")

        for article in articles:
            href = article.find("a")["href"]
            match = re.search(r"/archives/(\d+)\.html", href)
            if match:
                ids.append(match.group(1))
        print(f"Page {page} is proccessed.")
        time.sleep(0.5)
    return ids


def get_ids(filename: str) -> list[str]:
    with open(filename, "r") as f:
        data = json.load(f)
    # support both plain array and object-with-first-array
    if isinstance(data, list):
        return data
    # object: return the first array value found
    for v in data.values():
        if isinstance(v, list):
            return v
    raise ValueError("No array found in JSON")


def get_threads(ids: list[str]):
    out_dir = Path(__file__).parents[2] / "data" / "threads"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(ids)
    fetched = 0
    for n, id_ in enumerate(ids, 1):
        out_file = out_dir / f"thread_{id_}.html"
        if out_file.exists():
            status = "skip"
        else:
            url = CONTENT_PAGE_API.format(id=id_)
            try:
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                out_file.write_text(res.text, encoding="utf-8")
                status = "done"
            except requests.RequestException as e:
                status = f"fail: {e}"
            time.sleep(random.uniform(0.8, 2.0))
            fetched += 1

            if fetched % 100 == 0 and n < total:
                print(f"\n--- 休息 30 秒 (已爬 {fetched} 篇) ---", flush=True)
                time.sleep(30)

        print(f"\r({n}/{total}) [{status}] {id_}".ljust(80), end="", flush=True)

    print()


def export_as_json(thing: any, filename: str):
    with open(filename, "w") as f:
        json.dump(thing, f)


# Check whether given json contain an array and all element in array is unique
def exam_unique(filename: str):
    with open(filename, "r") as f:
        ids = json.load(f)

    total = len(ids)
    unique_ids = list(dict.fromkeys(ids))
    unique_count = len(unique_ids)
    duplicate_count = total - unique_count

    if duplicate_count == 0:
        print(f"✓ All {total} IDs are unique.")
    else:
        print(
            f"✗ Found {duplicate_count} duplicate(s) out of {total} IDs ({unique_count} unique)."
        )
        seen = set()
        for id_ in ids:
            if id_ in seen:
                print(f"  Duplicate ID: {id_}")
            else:
                seen.add(id_)

    return unique_ids


def _get_last_page(soup: BeautifulSoup) -> int:
    pagination = soup.find("ul", class_="pagination")
    if pagination is None:
        return 1
    items = pagination.find_all("li")
    if len(items) < 2:
        return 1
    try:
        return int(items[-2].get_text(strip=True))
    except ValueError:
        return 1


def _parse_thread_items(soup: BeautifulSoup, date_str: str) -> list[dict]:
    main = soup.find(id="mainThread")
    if main is None:
        return []
    records = []
    for a in main.find_all("a"):
        href = a.get("href", "")
        match = re.search(r"/board/(\d+)/", href)
        if not match:
            continue
        thread_id = match.group(1)
        title_tag = a.find("span", class_="title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        records.append({"id": thread_id, "title": title, "created_at": date_str})
    return records


def fetch_thread_index(
    start: str = "2026-04-01",
    end: str = "2026-04-25",
    output: str | None = None,
) -> None:
    if output is None:
        project_root = Path(__file__).parents[2]
        output = str(project_root / "data" / "thread_index_from_04_01_to_04_25.csv")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "created_at"])
        if write_header:
            writer.writeheader()

        current = start_date
        while current <= end_date:
            date_str = current.isoformat()
            last_page = 1
            page = 1

            while True:
                url = KAKOLOG_PAGE_API.format(date=date_str, page=page)
                print(f"\r{date_str} page:{page}".ljust(40), end="", flush=True)

                try:
                    res = requests.get(url, headers=headers, timeout=10)
                except requests.RequestException as e:
                    print(f"\n  {date_str} page:{page} error: {e}")
                    break

                if res.history:
                    print(f"\n  {date_str} → redirected, skip")
                    break

                soup = BeautifulSoup(res.text, "html.parser")
                records = _parse_thread_items(soup, date_str)
                writer.writerows(records)
                f.flush()

                last_page = _get_last_page(soup)
                if page >= last_page:
                    break

                page += 1
                time.sleep(random.uniform(1.0, 1.5))

            print(f"\r{date_str} done ({last_page} pages)".ljust(40))
            current += timedelta(days=1)
            time.sleep(random.uniform(1.0, 2.0))


def analyze_thread_index(filepath: str) -> None:
    counts: dict[str, int] = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["created_at"]
            counts[d] = counts.get(d, 0) + 1

    total = sum(counts.values())
    values = list(counts.values())

    print(f"總串數: {total}")
    print(f"日期數: {len(counts)}")
    print()
    print("每日串數統計:")
    print(f"  平均:   {statistics.mean(values):.1f}")
    print(f"  中位數: {statistics.median(values):.1f}")
    print(f"  最小:   {min(values)}")
    print(f"  最大:   {max(values)}")
    print()
    print(f"{'日期':<12} {'串數':>6}")
    print("-" * 20)
    for date_str in sorted(counts):
        print(f"{date_str:<12} {counts[date_str]:>6}")


def sample_thread_index_by_created_at(input: str, output: str, sample_each_date: int):
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    with open(input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["created_at"]].append(row)

    sampled: list[dict] = []
    for date_str in sorted(groups):
        group = groups[date_str]
        k = min(sample_each_date, len(group))
        sampled.extend(random.sample(group, k))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "created_at", "fetched"])
        writer.writeheader()
        for row in sampled:
            writer.writerow(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "fetched": "",
                }
            )


def fetch_threads_from_index(index_filepath: str) -> None:
    index_path = Path(index_filepath)
    raw_dir = Path(__file__).parents[2] / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(index_path, encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1  # subtract header row

    fetched_this_session = 0
    with open(index_path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), 1):
            out_file = raw_dir / f"thread_{row['id']}.html"

            if out_file.exists():
                print(
                    f"\r({n}/{total}) [skip] {row['id']}".ljust(80), end="", flush=True
                )
                continue

            tmp_file = out_file.with_suffix(".tmp")
            url = BOARD_PAGE_API.format(id=row["id"])
            try:
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                tmp_file.write_text(res.text, encoding="utf-8")
                tmp_file.replace(out_file)
                status = "done"
            except requests.RequestException as e:
                status = f"fail: {e}"

            print(
                f"\r({n}/{total}) [{status}] {row['id']}".ljust(80), end="", flush=True
            )
            time.sleep(random.uniform(1.0, 1.5))
            fetched_this_session += 1
            if (fetched_this_session % 1000) == 0:
                time.sleep(60 * 15)

    print()


if __name__ == "__main__":
    ## Scrape
    # fetch_thread_index(
    #     start="2026-04-01",
    #     end="2026-04-25",
    #     output="data/thread_index_from_04_01_to_04_25.csv",
    # )

    ## Statistics
    # analyze_thread_index("data/thread_index_from_04_01_to_04_25.csv")

    # sample_thread_index_by_created_at(
    #     "data/thread_index_from_04_01_to_04_25.csv",
    #     "data/thread_index_sampled.csv",
    #     400,
    # )

    # analyze_thread_index("data/thread_index_sampled.csv")

    pass
fetch_threads_from_index("data/thread_index_sampled.csv")
