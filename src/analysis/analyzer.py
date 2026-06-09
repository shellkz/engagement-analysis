import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.post import Post
from src.models.thread import Thread


def load_threads(jsonl_path: str) -> list[Thread]:
    threads = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            posts = [Post(**p) for p in data.pop("posts")]
            threads.append(Thread(**data, posts=posts))
    return threads


def engagement_score(thread: Thread) -> float:
    post_count = len(thread.posts)
    reply_to_count = sum(1 for p in thread.posts if p.reply_to)
    return math.log(post_count + 1) + math.log(reply_to_count + 1)


def _parse_post_time(created_at: str) -> datetime:
    # Format: "2023/07/18(火) 20:54:02"
    cleaned = re.sub(r"\([^)]+\)", "", created_at).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M:%S")


def compute_reply_interval_percentile(
    threads: list[Thread], percentile: float
) -> float:
    intervals = []
    for thread in threads:
        try:
            times = [_parse_post_time(p.created_at) for p in thread.posts]
        except ValueError:
            continue
        for i in range(len(times) - 1):
            intervals.append((times[i + 1] - times[i]).total_seconds())
    return float(np.percentile(intervals, percentile))


# 至少有 (coverage * 100)% 的Thread擁有N篇以上的Post
def compute_N(threads: list[Thread], coverage: float = 0.8) -> int:
    """N such that `coverage` fraction of threads have at least N posts."""
    lengths = [len(t.posts) for t in threads]
    return max(2, int(np.percentile(lengths, (1 - coverage) * 100)))


def _get_early_posts(
    thread: Thread, N: int, mode: str, window_minutes: float
) -> list[Post]:
    if mode == "posts":
        return thread.posts[:N]
    if not thread.posts:
        return []
    try:
        start = _parse_post_time(thread.posts[0].created_at)
    except ValueError:
        return []
    cutoff = start + timedelta(minutes=window_minutes)
    result = []
    for p in thread.posts:
        try:
            if _parse_post_time(p.created_at) <= cutoff:
                result.append(p)
        except ValueError:
            continue
    return result


def _get_early_intervals(posts: list[Post]) -> list[float] | None:
    if len(posts) < 2:
        return None
    try:
        times = [_parse_post_time(p.created_at) for p in posts]
    except ValueError:
        return None
    return [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]


def _early_reply_interval_mean(posts: list[Post]) -> float | None:
    intervals = _get_early_intervals(posts)
    if intervals is None:
        return None
    return float(np.mean(intervals))


def _early_post_interval_variation(posts: list[Post]) -> float | None:
    intervals = _get_early_intervals(posts)
    if intervals is None or len(intervals) < 2:
        return None
    mean = float(np.mean(intervals))
    if mean == 0:
        return None
    return float(np.std(intervals) / mean)


def _early_interval_slope(posts: list[Post]) -> float | None:
    intervals = _get_early_intervals(posts)
    if intervals is None or len(intervals) < 2:
        return None
    x = np.arange(len(intervals), dtype=float)
    return float(np.polyfit(x, intervals, 1)[0])


def _early_refer_to_ratio(posts: list[Post]) -> float | None:
    if not posts:
        return None
    return sum(1 for p in posts if p.reply_to) / len(posts)


def _early_refer_to_count(posts: list[Post]) -> int | None:
    if not posts:
        return None
    return sum(1 for p in posts if p.reply_to)


def _early_image_count(posts: list[Post]) -> int | None:
    if not posts:
        return None
    return sum(len(p.image_urls) for p in posts)


def _early_post_count(posts: list[Post]) -> int:
    return len(posts)


def _early_max_reply_chain_depth(posts: list[Post]) -> int:
    depth: dict[int, int] = {}
    for p in sorted(posts, key=lambda p: p.post_num):
        if not p.reply_to:
            depth[p.post_num] = 0
        else:
            parent_depth = max(depth.get(ref, 0) for ref in p.reply_to)
            depth[p.post_num] = parent_depth + 1
    return max(depth.values(), default=0)


def _first_post_length(thread: Thread) -> int | None:
    if not thread.posts:
        return None
    return len(thread.posts[0].content)


# 至少有 (coverage * 100)% 的Thread在surge後擁有N篇以上的Post
def compute_N_surge(threads: list[Thread], T: float, coverage: float = 0.8) -> int:
    lengths_after_first_surge = []
    for thread in threads:
        try:
            times = [_parse_post_time(p.created_at) for p in thread.posts]
        except ValueError:
            continue
        for i in range(len(times) - 1):
            if (times[i + 1] - times[i]).total_seconds() < T:
                lengths_after_first_surge.append(len(thread.posts) - (i + 1))
                break
    if not lengths_after_first_surge:
        return 2
    return max(2, int(np.percentile(lengths_after_first_surge, (1 - coverage) * 100)))


# surge後N則中，(後半段平均間隔 / 前半段平均間隔)的比值
# > 1 後繼無力，< 1 加速，≈ 1 維持
def _early_growth_after_first_surge(thread: Thread, N: int, T: float) -> float | None:
    try:
        times = [_parse_post_time(p.created_at) for p in thread.posts]
    except ValueError:
        return None

    surge_idx = None
    for i in range(len(times) - 1):
        if (times[i + 1] - times[i]).total_seconds() < T:
            surge_idx = i + 1
            break

    if surge_idx is None:
        return None

    times_after_surge = times[surge_idx : surge_idx + N]
    if len(times_after_surge) < 4:
        return None

    intervals = [
        (times_after_surge[i + 1] - times_after_surge[i]).total_seconds()
        for i in range(len(times_after_surge) - 1)
    ]
    mid = len(intervals) // 2
    mean_first = float(np.mean(intervals[:mid]))
    mean_second = float(np.mean(intervals[mid:]))

    if mean_first == 0:
        return None
    return mean_second / mean_first


def build_features(
    threads: list[Thread],
    labels: list[int],
    N: int,
    N_surge: int,
    T: float,
    mode: str = "posts",
    window_minutes: float = 60.0,
    feature_set: str = "full",
) -> pd.DataFrame:
    records = []
    for thread, label in zip(threads, labels):
        posts = _get_early_posts(thread, N, mode, window_minutes)
        if feature_set == "baseline":
            record = {
                "thread_id": thread.thread_id,
                "early_post_count": _early_post_count(posts),
                "label": label,
            }
        else:
            record = {
                "thread_id": thread.thread_id,
                "early_post_interval_variation": _early_post_interval_variation(posts),
                "early_refer_to_ratio": _early_refer_to_ratio(posts),
                "early_image_count": _early_image_count(posts),
                # "early_interval_mean": _early_reply_interval_mean(posts),
                # "early_growth_after_first_surge": _early_growth_after_first_surge(thread, N_surge, T),
                # "early_interval_slope": _early_interval_slope(
                #     posts
                # ),
                # "early_refer_to_count": _early_refer_to_count(posts),
                # "early_max_reply_chain_depth": _early_max_reply_chain_depth(posts),  # 與early_refer_to_ratio共線，效果普通
                # "first_post_length": _first_post_length(thread),  # 假說E：加入後ROC-AUC下降，移除
                "label": label,
            }
        records.append(record)
    return pd.DataFrame(records)


def train(train_feature_table: pd.DataFrame) -> Pipeline:
    feature_cols = [
        c for c in train_feature_table.columns if c not in ("thread_id", "label")
    ]
    X = train_feature_table[feature_cols].dropna()
    y = train_feature_table.loc[X.index, "label"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced")),
        ]
    )
    model.fit(X, y)

    coefs = model.named_steps["lr"].coef_[0]
    print(pd.DataFrame({"feature": feature_cols, "coefficient": coefs}))
    return model


def predict(model: Pipeline, test_feature_table: pd.DataFrame) -> None:
    feature_cols = [
        c for c in test_feature_table.columns if c not in ("thread_id", "label")
    ]
    X = test_feature_table[feature_cols].dropna()
    y = test_feature_table.loc[X.index, "label"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"ROC-AUC : {roc_auc_score(y, y_prob):.3f}")
    print(f"F1      : {f1_score(y, y_pred):.3f}  (positive class = high engagement)")


if __name__ == "__main__":
    project_root = Path(__file__).parents[2]
    jsonl_path = project_root / "data" / "threads.jsonl"

    threads = load_threads(str(jsonl_path))

    lengths = [len(t.posts) for t in threads]
    print(f"thread 數量: {len(threads)}")
    print(
        f"post 數量  min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.0f}, mean={np.mean(lengths):.1f}"
    )

    scores = [engagement_score(t) for t in threads]
    threshold = np.percentile(scores, 80)
    labels = [1 if s >= threshold else 0 for s in scores]

    X_train, X_test, y_train, y_test = train_test_split(
        threads, labels, test_size=0.2, random_state=42
    )

    MODE = "time"  # "posts" or "time"
    WINDOW = 60.0
    N = 10
    T = compute_reply_interval_percentile(X_train, percentile=25)
    N_surge = 20

    print(
        f"mode = {MODE}, N = {N}, T = {T:.1f}s, N_surge = {N_surge}, window = {WINDOW}min"
    )
    lengths = [len(t.posts) for t in X_train]
    print(f"thread 長度 median={np.median(lengths):.0f}, mean={np.mean(lengths):.0f}")

    print("\n--- Baseline (early_post_count only) ---")
    df_train_bl = build_features(
        X_train,
        y_train,
        N,
        N_surge,
        T,
        mode=MODE,
        window_minutes=WINDOW,
        feature_set="baseline",
    )
    model_bl = train(df_train_bl)
    df_test_bl = build_features(
        X_test,
        y_test,
        N,
        N_surge,
        T,
        mode=MODE,
        window_minutes=WINDOW,
        feature_set="baseline",
    )
    predict(model_bl, df_test_bl)

    print("\n--- Full model ---")
    df_train = build_features(
        X_train, y_train, N, N_surge, T, mode=MODE, window_minutes=WINDOW
    )
    model = train(df_train)
    df_test = build_features(
        X_test, y_test, N, N_surge, T, mode=MODE, window_minutes=WINDOW
    )
    predict(model, df_test)
