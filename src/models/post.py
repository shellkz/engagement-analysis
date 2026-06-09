from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Post:
    post_num: int
    author: str
    created_at: str
    content: str
    crawled_at: str = ""
    reply_to: list[int] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    text_color: Optional[str] = None
    font_size: Optional[str] = None


@dataclass
class PostV2:
    post_number: int
    author: str
    created_at: Optional[datetime]
    like_count: int
    content: str  # 純文字內文，a.reslink 和純圖片 p 已排除

    quoted_local_links: list[str] = field(
        default_factory=list
    )  # blockquote.ogp, animanch.com 網域
    quoted_external_links: list[str] = field(
        default_factory=list
    )  # blockquote.ogp, 非 animanch.com
    quoted_twitter_urls: list[str] = field(
        default_factory=list
    )  # blockquote.twitter-tweet

    image_urls: list[str] = field(default_factory=list)  # a.thumb (含 p 內)

    mentioned_by: list[int] = field(
        default_factory=list
    )  # div.reply > a.reslink (這樓被哪幾樓引用)
    mentions: list[int] = field(default_factory=list)  # p > a.reslink (這樓引用哪幾樓)
