from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.models.post import Post, PostV2


@dataclass
class Thread:
    thread_id: str
    title: str
    source_url: str
    created_at: str
    crawled_at: str
    posts: list[Post] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class ThreadV2:
    thread_id: str
    title: str
    categories: list[str]    # breadcrumb 中間段，排除首頁和自己
    created_at: Optional[datetime]  # 取自第一篇 post 的 created_at
    posts: list[PostV2] = field(default_factory=list)
