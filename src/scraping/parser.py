import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from src.models.post import Post, PostV2
from src.models.thread import Thread, ThreadV2


def from_html(filepath: str) -> Thread:
    path = Path(filepath)
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # --- Thread metadata ---
    thread_id_match = re.search(r"(\d+)\.html$", path.name)
    thread_id = thread_id_match.group(1) if thread_id_match else path.stem

    og_title = soup.find("meta", property="og:title")
    title = (
        og_title["content"]
        if og_title
        else (
            soup.select_one("h1#singletitle a") or soup.title or soup.new_tag("x")
        ).get_text(strip=True)
    )

    og_url = soup.find("meta", property="og:url")
    source_url = og_url["content"] if og_url else ""

    pubdate = soup.select_one(".pubdate time")
    created_at = pubdate.get_text(strip=True) if pubdate else ""

    crawled_at = datetime.now().isoformat()

    categories = [a.get_text(strip=True) for a in soup.select("span.categories a")]
    tags = [a.get_text(strip=True) for a in soup.select("span.tags a")]

    # --- Posts ---
    posts = []
    for section in soup.select("section#introtext, section#maintext"):
        for res_div in section.find_all("div", class_="res", recursive=False):
            t_h = res_div.find("div", class_="t_h")
            if not t_h:
                continue

            resnum = t_h.find("span", class_="resnum")
            resname = t_h.find("span", class_="resname")
            resdate = t_h.find("span", class_="resdate")

            post_num = int(resnum.get_text(strip=True)) if resnum else 0
            author = resname.get_text(strip=True) if resname else ""
            post_date = resdate.get_text(strip=True) if resdate else ""

            # Images from all resimg divs in this post (inside or outside t_b)
            image_urls = []
            for img_div in res_div.find_all("div", class_="resimg"):
                a_tag = img_div.find("a")
                if a_tag and a_tag.get("href"):
                    image_urls.append(a_tag["href"])

            t_b = res_div.find("div", class_="t_b")
            content = ""
            text_color = None
            font_size = None
            reply_to = []

            if t_b:
                style = t_b.get("style", "")
                if style:
                    m = re.search(r"color:\s*([^;]+)", style)
                    if m:
                        text_color = m.group(1).strip()
                    m = re.search(r"font-size:\s*([^;]+)", style)
                    if m:
                        font_size = m.group(1).strip()

                reply_to_set = set()
                for anchor in t_b.find_all("span", class_="anchor"):
                    m = re.search(r">>(\d+)", anchor.get_text())
                    if m:
                        reply_to_set.add(int(m.group(1)))
                reply_to = sorted(reply_to_set)

                # Remove resimg inside t_b so it doesn't appear in content text
                for img_div in t_b.find_all("div", class_="resimg"):
                    img_div.decompose()

                content = t_b.get_text(separator="\n").strip()

            posts.append(
                Post(
                    post_num=post_num,
                    author=author,
                    created_at=post_date,
                    content=content,
                    crawled_at=crawled_at,
                    reply_to=reply_to,
                    image_urls=image_urls,
                    text_color=text_color,
                    font_size=font_size,
                )
            )

    posts.sort(key=lambda p: re.sub(r"\([^)]+\)", "", p.created_at).strip())

    return Thread(
        thread_id=thread_id,
        title=title,
        source_url=source_url,
        created_at=created_at,
        crawled_at=crawled_at,
        posts=posts,
        categories=categories,
        tags=tags,
    )


def from_directory(directory: str) -> None:
    dir_path = Path(directory)
    html_files = sorted(dir_path.glob("*.html"))
    out_path = dir_path.parent / "threads.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, html_file in enumerate(html_files, 1):
            try:
                thread = from_html(str(html_file))
                f.write(
                    json.dumps(dataclasses.asdict(thread), ensure_ascii=False) + "\n"
                )
                print(
                    f"\r({i}/{len(html_files)}) {html_file.name}".ljust(60),
                    end="",
                    flush=True,
                )
            except Exception as e:
                print(f"\r({i}/{len(html_files)}) [FAIL] {html_file.name}: {e}")

    print(f"\nDone → {out_path}")


_LOCAL_DOMAINS = ("https://bbs.animanch.com/", "https://animanch.com/")


def _parse_post_datetime(s: str) -> datetime:
    # "26/05/31(日) 21:00:52" → datetime
    cleaned = re.sub(r"\([^)]+\)", "", s).strip()
    return datetime.strptime(cleaned, "%y/%m/%d %H:%M:%S")


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(repr(obj))


def from_html_v2(filepath: str) -> ThreadV2:
    path = Path(filepath)
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Thread ID from canonical URL
    canonical = soup.find("link", rel="canonical")
    if canonical:
        m = re.search(r"/board/(\d+)/", canonical.get("href", ""))
        thread_id = m.group(1) if m else path.stem
    else:
        m = re.search(r"(\d+)", path.stem)
        thread_id = m.group(1) if m else path.stem

    # Title: h1#threadTitle, fallback to og:title
    h1 = soup.find("h1", id="threadTitle")
    if h1:
        title = h1.get_text(strip=True)
    else:
        og = soup.find("meta", property="og:title")
        title = (
            og["content"]
            if og
            else (soup.title.get_text(strip=True) if soup.title else "")
        )

    # Categories: breadcrumb spans, exclude first (home) and last (self)
    breadcrumb = soup.find(id="breadcrumb")
    if breadcrumb:
        spans = breadcrumb.find_all("span")
        categories = [s.get_text(strip=True) for s in spans[1:-1]]
    else:
        categories = []

    posts = []
    res_list = soup.find("ul", id="resList")
    if res_list:
        for li in res_list.find_all("li", id=re.compile(r"^res\d+")):
            divs = li.find_all("div", recursive=False)
            if len(divs) < 2:
                continue
            header_div, body_div = divs[0], divs[1]

            # Rule 1-4: header fields
            resnumber = header_div.find("span", class_="resnumber")
            resname = header_div.find("span", class_="resname")
            resposted = header_div.find("span", class_="resposted")
            vcount = header_div.find("span", class_="vcount")
            post_number = int(resnumber.get_text(strip=True)) if resnumber else 0
            author = resname.get_text(strip=True) if resname else ""
            created_at_raw = resposted.get_text(strip=True) if resposted else ""
            created_at = (
                _parse_post_datetime(created_at_raw) if created_at_raw else None
            )
            like_count = (
                int(
                    vcount.get_text(strip=True)
                    if vcount.get_text(strip=True).isdigit()
                    else "0"
                )
                if vcount
                else 0
            )

            # Rule 5: content — direct p and br children, skip a.reslink and image-only p
            content_parts = []
            for child in body_div.children:
                if child.name == "p":
                    line_parts = []
                    for node in child.children:
                        if node.name is None:
                            if str(node).strip():
                                line_parts.append(str(node))
                        elif node.name == "br":
                            line_parts.append("\n")
                        elif node.name == "a" and "reslink" in node.get("class", []):
                            pass
                        elif node.name == "a" and "thumb" in node.get("class", []):
                            pass
                        else:
                            line_parts.append(node.get_text())
                    line = "".join(line_parts).strip()
                    if line:
                        content_parts.append(line + "\n")
                elif child.name == "br":
                    content_parts.append("\n")
            content = "".join(content_parts).strip()

            # Rule 6: quoted thread links — blockquote.ogp only, split local/external
            quoted_local_links: list[str] = []
            quoted_external_links: list[str] = []
            for bq in body_div.find_all("blockquote", class_="ogp"):
                for a in bq.find_all("a"):
                    href = a.get("href", "")
                    if href:
                        if any(href.startswith(d) for d in _LOCAL_DOMAINS):
                            quoted_local_links.append(href)
                        else:
                            quoted_external_links.append(href)

            # Rule 7: image URLs — direct a.thumb + p > a.thumb
            image_urls: list[str] = []
            for child in body_div.children:
                if child.name == "a" and "thumb" in child.get("class", []):
                    href = child.get("href", "")
                    if href:
                        image_urls.append(href)
                elif child.name == "p":
                    for a in child.find_all("a", class_="thumb"):
                        href = a.get("href", "")
                        if href:
                            image_urls.append(href)

            # Rule 8: Twitter URLs — blockquote.twitter-tweet, last <a>
            quoted_twitter_urls: list[str] = []
            for bq in body_div.find_all("blockquote", class_="twitter-tweet"):
                links = bq.find_all("a")
                if links:
                    href = links[-1].get("href", "")
                    if href:
                        quoted_twitter_urls.append(href)

            # Rule 9: mentioned_by — div.reply > a.reslink
            mentioned_by: list[int] = []
            reply_div = body_div.find("div", class_="reply")
            if reply_div:
                for a in reply_div.find_all("a", class_="reslink"):
                    m = re.search(r"#res(\d+)", a.get("href", ""))
                    if m:
                        mentioned_by.append(int(m.group(1)))

            # Rule 10: mentions — direct p > a.reslink, deduped, ordered
            mentions: list[int] = []
            seen: set[int] = set()
            for p in body_div.find_all("p", recursive=False):
                for a in p.find_all("a", class_="reslink"):
                    m = re.search(r"#res(\d+)", a.get("href", ""))
                    if m:
                        num = int(m.group(1))
                        if num not in seen:
                            seen.add(num)
                            mentions.append(num)

            posts.append(
                PostV2(
                    post_number=post_number,
                    author=author,
                    created_at=created_at,
                    like_count=like_count,
                    content=content,
                    quoted_local_links=quoted_local_links,
                    quoted_external_links=quoted_external_links,
                    image_urls=image_urls,
                    quoted_twitter_urls=quoted_twitter_urls,
                    mentioned_by=mentioned_by,
                    mentions=mentions,
                )
            )

    return ThreadV2(
        thread_id=thread_id,
        title=title,
        categories=categories,
        created_at=posts[0].created_at if posts else None,
        posts=posts,
    )


def from_directory_v2(source_directory: str, output_filename: str) -> None:
    src = Path(source_directory)
    html_files = sorted(src.glob("*.html"))
    total = len(html_files)

    out_path = Path(output_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, html_file in enumerate(html_files, 1):
            try:
                thread = from_html_v2(str(html_file))
                f.write(
                    json.dumps(
                        dataclasses.asdict(thread),
                        ensure_ascii=False,
                        default=_json_default,
                    )
                    + "\n"
                )
                print(f"\r({i}/{total}) {html_file.name}".ljust(60), end="", flush=True)
            except Exception as e:
                print(f"\r({i}/{total}) [FAIL] {html_file.name}: {e}")

    print(f"\nDone → {out_path}")


# # Parse html from data/threads/ as one big threads.jsonl
# # Each row is a single thread(contain posts)
# if __name__ == "__main__":
#     project_root = Path(__file__).parents[2]
#     from_directory(str(project_root / "data" / "threads"))

if __name__ == "__main__":
    # thread = from_html_v2("data/raw/thread_6551248.html")

    from_directory_v2("data/raw", "data/parsed/threads.jsonl")
