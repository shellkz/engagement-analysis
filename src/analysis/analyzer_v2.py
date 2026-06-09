import json
import math
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from src.models.post import PostV2
from src.models.thread import ThreadV2


# Implement
def _parse_dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _stream_threads(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            posts = [
                PostV2(**{**p, "created_at": _parse_dt(p["created_at"])})
                for p in data.pop("posts")
            ]
            data["created_at"] = _parse_dt(data.get("created_at"))
            yield ThreadV2(**data, posts=posts)


# Experiment Settings
## Label
def engagement_score(thread: ThreadV2) -> float:
    post_count = len(thread.posts)
    reply_count = sum(1 for p in thread.posts if p.mentions)
    return math.log(post_count + 1) + math.log(reply_count + 1)
    # return math.log(post_count + 1)


## Early constraint
def filter_early_posts(thread: ThreadV2) -> list[PostV2]:
    if not thread.created_at:
        return []
    early_cutoff = thread.created_at + timedelta(minutes=INITIAL_TIME_WINDOW_MINUTES)
    return [
        p
        for p in thread.posts
        if p.created_at is not None and p.created_at <= early_cutoff
    ]


HIGH_ENGAGEMENT_THRESHOLD = 80
TRAIN_RATIO = 0.8
INITIAL_TIME_WINDOW_MINUTES = 10


## Features Helper


def _get_early_intervals(thread: ThreadV2) -> list[float] | None:
    posts = filter_early_posts(thread)
    times = [p.created_at for p in posts if p.created_at is not None]
    if len(times) < 2:
        return None
    return [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]


## Features
def _early_post_interval_mean(thread: ThreadV2) -> float | None:
    intervals = _get_early_intervals(thread)
    if intervals is None:
        return None
    return float(np.mean(intervals))


def _early_post_interval_variation(thread: ThreadV2) -> float | None:
    intervals = _get_early_intervals(thread)
    if intervals is None or len(intervals) < 2:
        return None
    mean = float(np.mean(intervals))
    if mean == 0:
        return None
    return float(np.std(intervals) / mean)


def _early_post_mention_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(1 for p in posts if p.mentions) / len(posts)


def _early_post_mention_sum(thread: ThreadV2) -> int | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(len(p.mentions) for p in posts if p.mentions)


def _early_post_mention_mean(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return _early_post_mention_sum(thread) / len(posts)


def _early_post_mention_tree(thread: ThreadV2) -> dict[int, int]:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    depth: dict[int, int] = {}
    for p in sorted(posts, key=lambda p: p.post_number):
        if not p.mentions:
            depth[p.post_number] = 0
        else:
            parent_depth = max(depth.get(ref, 0) for ref in p.mentions)
            depth[p.post_number] = parent_depth + 1
    return depth


# Mention tree max depth
def _early_post_mention_depth_max(thread: ThreadV2) -> int | None:
    tree = _early_post_mention_tree(thread)
    return max(tree.values(), default=0)


# Mention tree mean depth
def _early_post_mention_depth_mean(thread: ThreadV2) -> int | None:
    tree = _early_post_mention_tree(thread)
    return sum(tree.values()) / len(tree.values())


# --- Width helpers ---


def _build_mention_children(posts: list[PostV2]) -> dict[int, list[int]]:
    """children[B] = early posts that directly mention B (B is the parent in reply tree)"""
    children: dict[int, list[int]] = {p.post_number: [] for p in posts}
    for p in posts:
        for ref in p.mentions:
            if ref in children:
                children[ref].append(p.post_number)
    return children


def _subtree_nodes(root: int, children: dict[int, list[int]]) -> set[int]:
    visited, queue = {root}, [root]
    for node in queue:
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return visited


def _largest_mention_subtree(posts: list[PostV2]) -> set[int]:
    """Largest reply subtree by node count among all root posts in the early window"""
    children = _build_mention_children(posts)
    all_children = {c for cs in children.values() for c in cs}
    roots = [pn for pn in children if pn not in all_children]
    if not roots:
        roots = [max(children, key=lambda k: len(children[k]))]
    return max((_subtree_nodes(r, children) for r in roots), key=len, default=set())


# Width 1 (narrowest): max direct in-degree — the single most-mentioned post
def _early_post_mention_width_max(thread: ThreadV2) -> int | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    children = _build_mention_children(posts)
    return max((len(v) for v in children.values()), default=0)


# Width 2: leaf count of the largest mention subtree — how dispersed the discussion is
def _early_post_mention_width_tree_leaves(thread: ThreadV2) -> int | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    children = _build_mention_children(posts)
    largest = _largest_mention_subtree(posts)
    return sum(1 for node in largest if not children.get(node))


# Width 3 (broadest): node count of the largest mention subtree — total discussion volume
def _early_post_mention_width_tree_nodes(thread: ThreadV2) -> int | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return len(_largest_mention_subtree(posts))


def _early_post_content_length_mean(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return float(np.mean([len(p.content) for p in posts]))


def _early_post_content_length_std(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if len(posts) < 2:
        return None
    return float(np.std([len(p.content) for p in posts]))


def _thread_title_length(thread: ThreadV2) -> int:
    return len(thread.title)


def _early_post_external_link_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(1 for p in posts if p.quoted_external_links) / len(posts)


def _early_post_local_link_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(1 for p in posts if p.quoted_local_links) / len(posts)


def _early_post_twitter_link_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(1 for p in posts if p.quoted_twitter_urls) / len(posts)


def _early_post_image_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(1 for p in posts if p.image_urls) / len(posts)


def _early_post_text_only_ratio(thread: ThreadV2) -> float | None:
    posts = filter_early_posts(thread)
    if not posts:
        return None
    return sum(
        1
        for p in posts
        if not p.image_urls
        and not p.quoted_twitter_urls
        and not p.quoted_external_links
        and not p.quoted_local_links
    ) / len(posts)


def _early_post_sum(thread: ThreadV2) -> int | None:
    return len(filter_early_posts(thread))


def build_baseline_features(thread: ThreadV2) -> dict:
    return {"_early_post_sum": _early_post_sum(thread)}


def build_custom_features(thread: ThreadV2) -> dict:
    return {
        "_early_post_sum": _early_post_sum(thread),
        #
        # "_early_post_interval_variation": _early_post_interval_variation(thread),
        # "_early_post_interval_mean": _early_post_interval_mean(thread),
        # #
        "early_post_mention_ratio": _early_post_mention_ratio(thread),
        "early_post_mention_mean": _early_post_mention_mean(thread),
        "early_post_mention_sum": _early_post_mention_sum(thread),
        #
        "early_post_mention_depth_max": _early_post_mention_depth_max(thread),
        "early_post_mention_depth_mean": _early_post_mention_depth_mean(thread),
        "early_post_mention_width_max": _early_post_mention_width_max(thread),
        # "early_post_mention_width_tree_leaves": _early_post_mention_width_tree_leaves(
        #     thread
        # ),
        # "early_post_mention_width_tree_nodes": _early_post_mention_width_tree_nodes(
        #     thread
        # ),
        #
        "early_post_external_link_ratio": _early_post_external_link_ratio(thread),
        "early_post_local_link_ratio": _early_post_local_link_ratio(thread),
        "early_post_twitter_link_ratio": _early_post_twitter_link_ratio(thread),
        #
        "early_post_image_ratio": _early_post_image_ratio(thread),
        "early_post_text_only_ratio": _early_post_text_only_ratio(thread),
        # #
        "early_post_content_length_mean": _early_post_content_length_mean(thread),
        "early_post_content_length_std": _early_post_content_length_std(thread),
        "thread_title_length": _thread_title_length(thread),
    }


# Access Point
def build_bunch(
    filepath: str, feature_factory: Callable[[ThreadV2], dict] = build_custom_features
) -> Bunch:
    feature_rows: list[dict] = []
    scores: list[float] = []
    created_at: list[datetime | None] = []

    # Features(data)
    for thread in _stream_threads(filepath):
        scores.append(engagement_score(thread))
        feature_rows.append(feature_factory(thread))
        created_at.append(thread.created_at)
    # Label(target)
    scores_arr = np.array(scores)
    threshold = float(np.percentile(scores_arr, HIGH_ENGAGEMENT_THRESHOLD))
    target = (scores_arr >= threshold).astype(int)
    data = pd.DataFrame(feature_rows)

    return Bunch(
        data=data.to_numpy(),
        target=target,
        feature_names=list(data.columns),
        scores=scores_arr,
        threshold=threshold,
        created_at=np.array(created_at),
    )


def train(feature: np.ndarray, label: np.ndarray, feature_names: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced")),
        ]
    )
    model.fit(feature, label)

    # Feature Info
    coefs = model.named_steps["lr"].coef_[0]
    print(
        pd.DataFrame({"feature": feature_names, "coefficient": coefs}).to_string(
            index=False
        )
    )
    return model


def train_lgbm(feature: np.ndarray, label: np.ndarray, feature_names: list[str]):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        class_weight="balanced", n_estimators=200, learning_rate=0.05
    )
    model.fit(feature, label)

    print(
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .to_string(index=False)
    )
    return model


def predict(model: Pipeline, test_feature: np.ndarray, test_label: np.ndarray) -> None:
    y_pred = model.predict(test_feature)
    y_prob = model.predict_proba(test_feature)[:, 1]
    print(f"ROC-AUC : {roc_auc_score(test_label, y_prob):.3f}")
    print(f"F1      : {f1_score(test_label, y_pred):.3f}")


def split_train_test(
    bunch: Bunch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Split train/test with older TRAIN_RATIO and newer  (1 - TRAIN_RATIO)
    timestamps = np.array(
        [dt.timestamp() if dt is not None else 0.0 for dt in bunch.created_at]
    )
    order = np.argsort(timestamps, kind="stable")
    cutoff = int(len(order) * TRAIN_RATIO)
    train_idx, test_idx = order[:cutoff], order[cutoff:]
    return (
        bunch.data[train_idx],
        bunch.target[train_idx],
        bunch.data[test_idx],
        bunch.target[test_idx],
    )


if __name__ == "__main__":
    data_path = "data/parsed/threads.jsonl"

    ## Statics
    # print("--- Early Window Coverage ---")
    # first_n_minutes = [5, 10, 15, 30, 60]
    # threads_for_coverage = [t for t in _stream_threads(data_path) if t.posts and t.created_at]
    # print("| window | p25   | p50   | p75   |")
    # print("|--------|-------|-------|-------|")
    # for w in first_n_minutes:
    #     cutoff_delta = timedelta(minutes=w)
    #     ratios = [
    #         sum(1 for p in t.posts if p.created_at and p.created_at <= t.created_at + cutoff_delta) / len(t.posts)
    #         for t in threads_for_coverage
    #     ]
    #     arr = np.array(ratios)
    #     p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    #     print(f"| {w:>5}m | {p25:.3f} | {p50:.3f} | {p75:.3f} |")

    # Experiment
    print("--- Baseline ---")
    bunch_baseline = build_bunch(data_path, build_baseline_features)
    train_X_bl, train_y_bl, test_X_bl, test_y_bl = split_train_test(bunch_baseline)
    model_bl = train(train_X_bl, train_y_bl, bunch_baseline.feature_names)
    predict(model_bl, test_X_bl, test_y_bl)

    print("\n--- Custom Logistic Reggression ---")
    bunch_custom = build_bunch(data_path, build_custom_features)
    train_X_cu, train_y_cu, test_X_cu, test_y_cu = split_train_test(bunch_custom)
    model_cu = train(train_X_cu, train_y_cu, bunch_custom.feature_names)
    predict(model_cu, test_X_cu, test_y_cu)

    print("\n--- Custom LightGBM ---")
    model_lgbm = train_lgbm(train_X_cu, train_y_cu, bunch_custom.feature_names)
    predict(
        model_lgbm,
        pd.DataFrame(test_X_cu, columns=bunch_custom.feature_names),
        test_y_cu,
    )
