#!/usr/bin/env python3
"""
Self-confidence V5: single-objective AI-likeness self-assessment experiment.

Three matched conditions:
  1) live  - generate prose and append <AI SCORE: n> after every sentence.
  2) post  - re-score the exact completed live text with the same original query.
  3) replay - for a deterministic question-level subset, replay sentence 1..N
              sequentially, using only earlier scores from that replay.

Default transport: OpenRouter chat/completions.
State: resumable SQLite/WAL.
Python: standard library only.

Example adaptive confirmatory run:
  python self_confidence_v5_7_promptbank.py --seed 20260906 \
    --min-cases 30 --batch-size 10 --max-cases 120 \
    --out self_confidence_v5_adaptive --replay-fraction 0.10 --dry-run

Then remove --dry-run to spend API credits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CODE_VERSION = "5.7"
PROMPT_VERSION = "ai-likeness-single-objective-v5.2"
REPLAY_VERSION = "question-level-full-sequential-replay-v1"
SCHEMA_VERSION = "5.3"
EXTENSION_VERSION = "expanded-replay-and-score-guided-correction-v2"
CASE_SENTENCE_MIN = 5
CASE_SENTENCE_MAX = 8

SYSTEM_PROMPT = (
    "Follow the user's writing task and formatting instructions exactly. "
    "Return only the requested output."
)

# EXACTLY this instruction is embedded in all three conditions.
AI_SCORE_INSTRUCTION = (
    "After every sentence, assess how likely an average human reader would be "
    "to think that sentence was AI-generated. Use an integer from 0 to 100, "
    "where 0 means very human-like and 100 means obviously AI-generated. Put "
    "the sentence's normal closing punctuation first, then immediately append "
    "the assessment as <AI SCORE: n>."
)

SLOT = "<AI SCORE: ?>"
# Accept a harmless parenthesized numeric variant such as <AI SCORE: (25)>,
# but reject nonnumeric, negative, or otherwise malformed score-like tags.
_SCORE_RE = re.compile(
    r"<\s*AI\s+SCORE\s*:\s*\(?\s*(\d{1,3})\s*\)?\s*>", re.IGNORECASE
)
_ANY_SCORE_RE = re.compile(
    r"<\s*AI\s+SCORE\s*:\s*(?:\(?\s*\d{1,3}\s*\)?|\?)\s*>",
    re.IGNORECASE,
)
_BARE_SCORE_RE = re.compile(
    r"(?<!<)\bAI\s+SCORE\s*:\s*\(?\s*(\d{1,3})\s*\)?\b",
    re.IGNORECASE,
)
_SCORE_LIKE_RE = re.compile(r"<\s*AI\s+SCORE\b", re.IGNORECASE)
_SCORE_BEFORE_TERMINAL_RE = re.compile(
    r"[ \t]*(<\s*AI\s+SCORE\s*:\s*\(?\s*\d{1,3}\s*\)?\s*>)([.!?…]+)",
    re.IGNORECASE,
)
_BARE_INTEGER_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*$")


# Exact model IDs recovered from the preceding V4 replay experiment are kept as
# aliases. OpenRouter route IDs are the current route names used by this runner.
MODEL_SPECS: dict[str, str] = {
    "gpt-5.5": "openai/gpt-5.5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash-0731",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "qwen3-30b-a3b-instruct-2507": "qwen/qwen3-30b-a3b-instruct-2507",
    "ministral-3-14b-2512": "mistralai/ministral-14b-2512",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}

# Keep provider choice as stable as practical across matched conditions.
# V5.4 showed Claude Platform on AWS cleanly serving LIVE/REPLAY while POST
# sometimes returned provider-level content_filter failures.
MODEL_PROVIDER_PREFERENCES: dict[str, Mapping[str, Any]] = {}


# Gemma 3 12B currently has only one OpenRouter provider (DeepInfra), and the
# V5.4 smoke test repeatedly hit 429s despite per-model concurrency=1. Space
# requests out rather than increasing concurrent pressure.
MODEL_MIN_INTERVAL_SECONDS: dict[str, float] = {
    "gemma-3-12b-it": 6.0,
}


@dataclass(frozen=True)
class Question:
    question_id: str
    family: str
    task: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ApiResult:
    text: str
    raw_json: Mapping[str, Any]
    provider: str | None
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class LiveParse:
    raw_text: str
    template: str
    scores: tuple[int, ...]
    units: tuple[str, ...]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_simple_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines without adding a python-dotenv dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def common_user_prefix(task: str) -> str:
    """
    Experimentally invariant prompt prefix.

    The exact original query and exact scoring instruction occur byte-for-byte
    in LIVE, POST, and every REPLAY step.
    """
    return (
        "ORIGINAL WRITING TASK\n"
        "---------------------\n"
        f"{task}\n\n"
        "SCORING INSTRUCTION\n"
        "-------------------\n"
        f"{AI_SCORE_INSTRUCTION}\n"
    )


def live_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                common_user_prefix(task)
                + "\nProduce the requested text now and follow the scoring "
                "instruction exactly."
            ),
        },
    ]


def post_messages(task: str, score_template: str) -> list[dict[str, str]]:
    expected_n = score_template.count(SLOT)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                common_user_prefix(task)
                + "\nThe requested text has already been produced below. "
                "Assess that exact completed text without rewriting it. There "
                f"are {expected_n} <AI SCORE: ?> placeholders, one for each "
                "sentence. Return exactly the same number of <AI SCORE: n> "
                "tags, one per line and in sentence order. Do not repeat, "
                "continue, or rewrite the text.\n\n"
                "COMPLETE TEXT\n"
                "-------------\n"
                f"{score_template}"
            ),
        },
    ]


def replay_messages(task: str, replay_prefix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                common_user_prefix(task)
                + "\nContinue the assessment for the text below. The only "
                "missing content is the single final <AI SCORE: ?>. Do not "
                "repeat, continue, or rewrite the text. Replace that placeholder "
                "with the score required by the scoring instruction and return "
                "only <AI SCORE: n>.\n\n"
                "TEXT SO FAR\n"
                "-----------\n"
                f"{replay_prefix}"
            ),
        },
    ]


def canonicalize_live_score_placement(text: str) -> str:
    """
    Canonicalize the common harmless form:
        Sentence <AI SCORE: 20>.
    to:
        Sentence. <AI SCORE: 20>

    Raw provider output is still stored separately. This changes only the
    analysis template and prevents terminal punctuation after the tag from
    being misclassified as an unscored extra sentence.
    """
    return _SCORE_BEFORE_TERMINAL_RE.sub(
        lambda m: f"{m.group(2)} {m.group(1)}",
        text,
    )


def _reject_unparsed_score_like_tags(text: str, *, context: str) -> None:
    """Reject score-like tags not recognized as valid numeric score tags."""
    scrubbed = _SCORE_RE.sub("", text)
    scrubbed = re.sub(
        r"<AI\s+SCORE\s*:\s*\?\s*>", "", scrubbed, flags=re.IGNORECASE
    )
    match = _SCORE_LIKE_RE.search(scrubbed)
    if match:
        snippet = scrubbed[match.start() : match.start() + 80].splitlines()[0]
        raise ValueError(
            f"{context} contains malformed/unrecognized AI score tag: {snippet!r}"
        )


def parse_live(
    text: str,
    expected_min_scores: int | None = None,
    expected_max_scores: int | None = None,
) -> LiveParse:
    canonical = canonicalize_live_score_placement(text)
    _reject_unparsed_score_like_tags(canonical, context="live response")
    matches = list(_SCORE_RE.finditer(canonical))
    if not matches:
        raise ValueError("live response contains no <AI SCORE: n> tags")

    scores: list[int] = []
    units: list[str] = []
    last = 0
    for match in matches:
        score = int(match.group(1))
        if not 0 <= score <= 100:
            raise ValueError(f"AI score outside 0..100: {score}")
        unit = canonical[last : match.start()]
        if not unit.strip():
            raise ValueError(
                "empty text unit before an AI score; detached score blocks "
                "cannot be aligned sentence-by-sentence"
            )
        scores.append(score)
        units.append(unit)
        last = match.end()

    if expected_min_scores is not None and len(scores) < expected_min_scores:
        raise ValueError(
            f"live response has only {len(scores)} aligned score tags; "
            f"expected at least {expected_min_scores}"
        )
    if expected_max_scores is not None and len(scores) > expected_max_scores:
        raise ValueError(
            f"live response has {len(scores)} aligned score tags; "
            f"expected at most {expected_max_scores}"
        )

    trailing = canonical[last:]
    if trailing.strip():
        raise ValueError(
            "live response has non-whitespace text after the final AI score; "
            "the final written material is therefore unscored"
        )

    return LiveParse(
        raw_text=text,
        template=_SCORE_RE.sub(SLOT, canonical),
        scores=tuple(scores),
        units=tuple(units),
    )


def normalize_scores_to_slots(text: str) -> str:
    return _ANY_SCORE_RE.sub(SLOT, text)


def prose_lexical_tokens(text: str) -> tuple[str, ...]:
    """
    Compare prose while ignoring only score tags, whitespace, and punctuation.

    This accepts harmless formatting drift in POST but still rejects lexical
    rewriting, insertion, deletion, or reordering.
    """
    no_scores = _ANY_SCORE_RE.sub("", text)
    no_scores = _BARE_SCORE_RE.sub("", no_scores)
    normalized = no_scores.replace("’", "'").replace("‘", "'")
    return tuple(
        token.casefold()
        for token in re.findall(r"\w+(?:'\w+)*", normalized, flags=re.UNICODE)
    )


def parse_post(text: str, expected_template: str) -> tuple[int, ...]:
    """
    Parse POST scores while tolerating formatting but not ambiguity.

    Accepted forms:
      * exactly N proper tags, including <AI SCORE: (25)>;
      * exactly N bare ``AI SCORE: 25`` labels;
      * exactly N integer-only nonempty lines (e.g. ``5\n4\n8``).

    Multiple answer sets, commentary containing additional scores, negative or
    nonnumeric score-like tags, and out-of-range values remain errors.
    """
    expected_n = expected_template.count(SLOT)
    _reject_unparsed_score_like_tags(text, context="post response")

    proper = [int(m.group(1)) for m in _SCORE_RE.finditer(text)]
    if proper:
        if len(proper) != expected_n:
            raise ValueError(
                f"post score count mismatch: got {len(proper)} proper tags, "
                f"expected {expected_n}"
            )
        if any(not 0 <= score <= 100 for score in proper):
            raise ValueError("post response contains a score outside 0..100")
        return tuple(proper)

    bare = [int(m.group(1)) for m in _BARE_SCORE_RE.finditer(text)]
    if bare:
        if len(bare) != expected_n:
            raise ValueError(
                f"post score count mismatch: got {len(bare)} bare labels, "
                f"expected {expected_n}"
            )
        if any(not 0 <= score <= 100 for score in bare):
            raise ValueError("post response contains a score outside 0..100")
        return tuple(bare)

    # Some providers obey the requested cardinality/order but omit the label.
    # Only accept if *every* nonempty non-fence line is a single integer.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and line.strip() not in {"```", "```text"}
    ]
    integer_scores: list[int] = []
    for line in lines:
        match = _BARE_INTEGER_LINE_RE.fullmatch(line)
        if not match:
            integer_scores = []
            break
        integer_scores.append(int(match.group(1)))

    if integer_scores:
        if len(integer_scores) != expected_n:
            raise ValueError(
                f"post score count mismatch: got {len(integer_scores)} "
                f"integer-only lines, expected {expected_n}"
            )
        if any(not 0 <= score <= 100 for score in integer_scores):
            raise ValueError("post response contains a score outside 0..100")
        return tuple(integer_scores)

    raise ValueError(
        f"post response contains no unambiguous sequence of {expected_n} scores"
    )


def parse_single_replay_score(
    text: str,
    previous_replay_scores: Sequence[int],
) -> tuple[int, str]:
    """
    Parse one replay target conservatively but tolerate harmless formatting.

    Accepted:
      * exactly one proper <AI SCORE: n> tag;
      * exactly one bare 'AI SCORE: n' if angle brackets were omitted;
      * a repeated full replay prefix whose earlier score tags exactly equal
        previous_replay_scores, followed by exactly one new final score.

    Ambiguous multiple candidate scores remain an error.
    """
    _reject_unparsed_score_like_tags(text, context="replay response")
    matches = list(_SCORE_RE.finditer(text))
    tagged_scores = [int(m.group(1)) for m in matches]
    if any(not 0 <= score <= 100 for score in tagged_scores):
        raise ValueError("replay response contains a score outside 0..100")

    if len(tagged_scores) == 1:
        remainder = _SCORE_RE.sub("", text).strip()
        return tagged_scores[0], (
            "exact_tag" if not remainder else "tag_with_extra_text"
        )

    if len(tagged_scores) > 1:
        prior = list(map(int, previous_replay_scores))
        if tagged_scores[:-1] == prior:
            return tagged_scores[-1], "repeated_prefix"
        raise ValueError(
            "replay response contains multiple ambiguous AI score tags: "
            f"{tagged_scores}; expected prior replay scores {prior}"
        )

    bare = [int(m.group(1)) for m in _BARE_SCORE_RE.finditer(text)]
    if len(bare) == 1 and 0 <= bare[0] <= 100:
        return bare[0], "bare_score"
    if bare:
        raise ValueError(f"replay response contains ambiguous bare scores: {bare}")
    raise ValueError("replay response contains no complete numeric AI score")


def split_template(template: str) -> list[str]:
    pieces = template.split(SLOT)
    if len(pieces) < 2:
        raise ValueError("template contains no score slots")
    return pieces


def build_replay_prefix(
    template: str,
    previous_replay_scores: Sequence[int],
    target_sentence_index: int,
) -> str:
    """
    Build text through target sentence, using ONLY prior replay scores.

    Deliberately has no parameter for live/post scores, making cross-condition
    score leakage difficult to introduce accidentally.
    """
    pieces = split_template(template)
    n_scores = len(pieces) - 1

    if not 0 <= target_sentence_index < n_scores:
        raise IndexError(target_sentence_index)
    if len(previous_replay_scores) != target_sentence_index:
        raise ValueError(
            "previous_replay_scores must contain exactly one score for each "
            "earlier sentence in this replay"
        )

    out: list[str] = [pieces[0]]
    for i in range(target_sentence_index):
        score = int(previous_replay_scores[i])
        if not 0 <= score <= 100:
            raise ValueError(f"invalid previous replay score: {score}")
        out.append(f"<AI SCORE: {score}>")
        out.append(pieces[i + 1])
    out.append(SLOT)
    return "".join(out)


def select_replay_questions(
    question_ids: Iterable[str], fraction: float, seed: int
) -> tuple[str, ...]:
    ids = sorted(set(question_ids))
    if not 0 <= fraction <= 1:
        raise ValueError("replay fraction must be in [0, 1]")
    if not ids or fraction == 0:
        return ()
    n = max(1, int(round(len(ids) * fraction)))
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(ids, min(n, len(ids)))))


# ---------------------------------------------------------------------------
# Deterministic narrative cases
# ---------------------------------------------------------------------------

FAMILY_NAMES = (
    "personal_social_post",
    "recommendation_post",
    "product_experience",
    "travel_moment",
    "workplace_anecdote",
    "neighborhood_observation",
    "complaint_with_story",
    "event_recap",
    "opinion_from_experience",
    "message_to_friends",
)

PEOPLE = (
    "my older sister",
    "a colleague",
    "my neighbor",
    "an old university friend",
    "my cousin",
    "a parent from school",
    "a former manager",
    "a friend I had not seen for years",
)

PLACES = (
    "a small neighborhood cafe",
    "a crowded train station",
    "a local park",
    "a seaside promenade",
    "a supermarket",
    "a hotel lobby",
    "a community event",
    "a quiet side street",
    "an airport",
    "a weekend market",
)

DETAILS = (
    "an unexpected five-minute delay",
    "a handwritten sign",
    "a surprisingly kind response",
    "a minor misunderstanding",
    "a sudden change of plans",
    "a badly timed phone call",
    "a small thing that made the day easier",
    "an awkward but funny exchange",
    "a detail everyone else seemed to ignore",
    "a choice I regretted almost immediately",
)

TOPICS = (
    "trying a new coffee place",
    "buying a household gadget",
    "taking a short weekend trip",
    "dealing with customer support",
    "attending a school event",
    "working from a different place for a day",
    "choosing between two services",
    "returning to a place after several years",
    "helping someone with a practical problem",
    "changing my mind about something ordinary",
)

PRODUCTS = (
    "a cordless vacuum",
    "a compact coffee machine",
    "a pair of walking shoes",
    "a noise-cancelling headset",
    "a carry-on suitcase",
    "a kitchen mixer",
    "a budget office chair",
    "a phone power bank",
)

EVENTS = (
    "a neighborhood festival",
    "a school performance",
    "a small professional meetup",
    "a family birthday",
    "an outdoor concert",
    "a local sports event",
    "a public lecture",
    "a community clean-up day",
)


def _task_for_family(
    family: str,
    *,
    person: str,
    place: str,
    detail: str,
    topic: str,
    product: str,
    event: str,
) -> str:
    base = (
        f"Write {CASE_SENTENCE_MIN}-{CASE_SENTENCE_MAX} sentences in English. "
        "Do not use bullet points. "
    )
    if family == "personal_social_post":
        return (
            base
            + f"Write a first-person social-media post about {topic}. Mention "
            f"{person}, {place}, and {detail}. The post should read as an ordinary "
            "personal update rather than an article."
        )
    if family == "recommendation_post":
        return (
            base
            + f"Write a first-person recommendation to friends after an experience "
            f"at {place}. Include {detail} and explain one concrete reason you "
            "would or would not recommend it."
        )
    if family == "product_experience":
        return (
            base
            + f"Write a first-person account of using {product} for several weeks. "
            f"Include one thing you liked, one annoyance, and {detail}. Do not make "
            "it sound like advertising copy."
        )
    if family == "travel_moment":
        return (
            base
            + f"Write a short first-person travel anecdote centered on {place}. "
            f"Include {person} and {detail}. Focus on one small moment rather than "
            "summarizing an entire trip."
        )
    if family == "workplace_anecdote":
        return (
            base
            + f"Write a first-person workplace anecdote involving {person}. The "
            f"situation should involve {topic} and {detail}. Keep it suitable for "
            "a casual professional social-media post."
        )
    if family == "neighborhood_observation":
        return (
            base
            + f"Write a first-person post about something you noticed around "
            f"{place}. Build the post around {detail} and a small interaction with "
            f"{person}."
        )
    if family == "complaint_with_story":
        return (
            base
            + f"Write a first-person complaint about {topic}, but tell it as a "
            f"specific story rather than a formal complaint letter. Include "
            f"{place} and {detail}."
        )
    if family == "event_recap":
        return (
            base
            + f"Write a first-person recap of {event}. Mention {person}, one "
            f"specific moment involving {detail}, and whether you would go again."
        )
    if family == "opinion_from_experience":
        return (
            base
            + f"Write a first-person opinion post about {topic}. Ground the opinion "
            f"in a concrete experience involving {place} and {detail}; do not turn "
            "it into a general essay."
        )
    if family == "message_to_friends":
        return (
            base
            + f"Write an informal update addressed to a group of friends about "
            f"{topic}. Mention {person}, {place}, and {detail}. It should feel like "
            "something a person might actually post to a private group."
        )
    raise KeyError(family)


def generate_questions(n: int, seed: int) -> list[Question]:
    if n < 1:
        raise ValueError("--cases must be >= 1")

    rng = random.Random(seed)
    questions: list[Question] = []
    used_tasks: set[str] = set()

    for i in range(n):
        family = FAMILY_NAMES[i % len(FAMILY_NAMES)]

        # Deterministic but varied. Retry combinations if a duplicate task occurs.
        for _ in range(100):
            values = {
                "person": rng.choice(PEOPLE),
                "place": rng.choice(PLACES),
                "detail": rng.choice(DETAILS),
                "topic": rng.choice(TOPICS),
                "product": rng.choice(PRODUCTS),
                "event": rng.choice(EVENTS),
            }
            task = _task_for_family(family, **values)
            if task not in used_tasks:
                break
        else:
            # This should only be reachable for extremely large synthetic runs.
            task += f" Use scenario variant {i + 1}."

        used_tasks.add(task)
        qid = f"q{i + 1:04d}_{family}"
        questions.append(
            Question(
                question_id=qid,
                family=family,
                task=task,
                metadata={"case_index": i, **values},
            )
        )

    # Shuffle case ordering without changing family balance or IDs.
    rng.shuffle(questions)
    return questions


PROMPT_BANK_VERSION = "prompts-v3"
PROMPT_BANK_EXPECTED_SHA256 = "be451428045e8b9fc2460d440ea2c962f3338237e87c3fa7dd6fff490569b5bf"
PRIMARY_SUBSET = "primary_english"
RUSSIAN_VALIDATION_SUBSET = "russian_validation"


def load_fixed_prompt_bank(path: Path, expected_sha256: str | None) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        raise FileNotFoundError(f"prompt bank not found: {path}")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"prompt bank SHA256 mismatch: expected {expected_sha256}, got {actual_sha}. "
            "Use the frozen prompts_v3.jsonl or explicitly pass the intended SHA."
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for key in ("prompt_id", "language", "subset", "domain", "full_task"):
            if key not in obj:
                raise ValueError(f"prompt bank line {line_no} missing {key!r}")
        pid = str(obj["prompt_id"])
        if pid in seen:
            raise ValueError(f"duplicate prompt_id {pid!r}")
        seen.add(pid)
        rows.append(dict(obj))
    if not rows:
        raise ValueError("prompt bank is empty")
    return rows, actual_sha


def balanced_primary_questions(
    bank_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[Question]:
    """
    Deterministic domain-balanced ordering of the frozen English primary bank.

    For the v3 bank (10 domains x 10 prompts), every consecutive batch of 10
    contains exactly one prompt from every domain. Only order is shuffled; prompt
    text and IDs remain immutable.
    """
    primary = [r for r in bank_rows if str(r.get("subset")) == PRIMARY_SUBSET]
    if not primary:
        raise ValueError(f"prompt bank has no {PRIMARY_SUBSET!r} rows")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in primary:
        if str(row.get("language")) != "en":
            raise ValueError(f"primary prompt {row['prompt_id']} is not English")
        groups.setdefault(str(row["domain"]), []).append(row)
    domains = sorted(groups)
    rng = random.Random(seed)
    for domain in domains:
        # Domain-specific deterministic shuffle prevents correlations with bank IDs.
        local = random.Random(
            int(hashlib.sha256(f"{seed}|{domain}".encode()).hexdigest()[:16], 16)
        )
        local.shuffle(groups[domain])
    # Stable randomized domain order, then round-robin one item per domain.
    rng.shuffle(domains)
    max_len = max(len(groups[d]) for d in domains)
    ordered_rows: list[Mapping[str, Any]] = []
    for slot in range(max_len):
        for domain in domains:
            if slot < len(groups[domain]):
                ordered_rows.append(groups[domain][slot])

    questions: list[Question] = []
    for i, row in enumerate(ordered_rows):
        questions.append(
            Question(
                question_id=str(row["prompt_id"]),
                family=str(row["domain"]),
                task=str(row["full_task"]),
                metadata={
                    "case_index": i,
                    "prompt_id": str(row["prompt_id"]),
                    "domain": str(row["domain"]),
                    "language": "en",
                    "subset": PRIMARY_SUBSET,
                    "bank_version": PROMPT_BANK_VERSION,
                },
            )
        )
    return questions


def russian_validation_questions(
    bank_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[Question]:
    rows = [r for r in bank_rows if str(r.get("subset")) == RUSSIAN_VALIDATION_SUBSET]
    if not rows:
        return []
    for row in rows:
        if str(row.get("language")) != "ru":
            raise ValueError(f"Russian validation prompt {row['prompt_id']} is not Russian")
    # Randomize presentation/order only; all ten are always used.
    rows = list(rows)
    local = random.Random(seed + 99173)
    local.shuffle(rows)
    return [
        Question(
            question_id=str(row["prompt_id"]),
            family=str(row["domain"]),
            task=str(row["full_task"]),
            metadata={
                "case_index": i,
                "validation_index": i,
                "prompt_id": str(row["prompt_id"]),
                "domain": str(row["domain"]),
                "language": "ru",
                "subset": RUSSIAN_VALIDATION_SUBSET,
                "bank_version": PROMPT_BANK_VERSION,
            },
        )
        for i, row in enumerate(rows)
    ]


def prompt_bank_summary(rows: Sequence[Mapping[str, Any]], actual_sha: str) -> dict[str, Any]:
    subsets: dict[str, int] = {}
    domains: dict[str, int] = {}
    languages: dict[str, int] = {}
    for r in rows:
        subsets[str(r["subset"])] = subsets.get(str(r["subset"]), 0) + 1
        domains[str(r["domain"])] = domains.get(str(r["domain"]), 0) + 1
        languages[str(r["language"])] = languages.get(str(r["language"]), 0) + 1
    return {
        "version": PROMPT_BANK_VERSION,
        "sha256": actual_sha,
        "n_prompts": len(rows),
        "subsets": dict(sorted(subsets.items())),
        "domains": dict(sorted(domains.items())),
        "languages": dict(sorted(languages.items())),
    }


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float,
        timeout_s: float,
        retries: int,
    ):
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries

    def call(
        self,
        *,
        model_id: str,
        model_route: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> ApiResult:
        payload = {
            "model": model_route,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        provider_pref = MODEL_PROVIDER_PREFERENCES.get(model_id)
        if provider_pref:
            payload["provider"] = dict(provider_pref)
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://example.invalid/anonymous-review",
                    "X-Title": "Self Confidence V5",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    raw_bytes = response.read()
                latency = time.perf_counter() - started
                data = json.loads(raw_bytes.decode("utf-8"))

                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("OpenRouter response has no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")

                # OpenRouter can return multipart content on some routes.
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    content = "".join(text_parts)

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "model returned empty final content "
                        f"(finish_reason={choices[0].get('finish_reason')!r})"
                    )

                usage = data.get("usage") or {}
                return ApiResult(
                    text=content,
                    raw_json=data,
                    provider=data.get("provider"),
                    latency_s=latency,
                    prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                    completion_tokens=_int_or_none(
                        usage.get("completion_tokens")
                    ),
                )

            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break

                retryable = True
                if isinstance(exc, urllib.error.HTTPError):
                    # Bad credentials / malformed request normally won't heal.
                    retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable:
                    break

                delay = min(90.0, 2.0 * (2 ** (attempt - 1)))
                if isinstance(exc, urllib.error.HTTPError):
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                delay += random.random() * 0.75
                time.sleep(delay)

        assert last_error is not None
        raise last_error


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=60.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def initialize(self) -> None:
        with self._schema_lock:
            with self.connect() as con:
                con.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS questions (
                        question_id TEXT PRIMARY KEY,
                        family TEXT NOT NULL,
                        task TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS cells (
                        model_id TEXT NOT NULL,
                        model_route TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        live_status TEXT NOT NULL DEFAULT 'pending',
                        live_text TEXT,
                        live_template TEXT,
                        live_error TEXT,
                        post_status TEXT NOT NULL DEFAULT 'pending',
                        post_text TEXT,
                        post_error TEXT,
                        PRIMARY KEY(model_id, question_id),
                        FOREIGN KEY(question_id) REFERENCES questions(question_id)
                    );

                    CREATE TABLE IF NOT EXISTS scores (
                        model_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        condition TEXT NOT NULL
                            CHECK(condition IN ('live','post','replay')),
                        sentence_index INTEGER NOT NULL,
                        score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 100),
                        PRIMARY KEY(
                            model_id, question_id, condition, sentence_index
                        )
                    );

                    CREATE TABLE IF NOT EXISTS api_calls (
                        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        sentence_index INTEGER,
                        prompt_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        provider TEXT,
                        latency_s REAL,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        response_text TEXT,
                        response_json TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS replay_selection (
                        question_id TEXT PRIMARY KEY,
                        selected INTEGER NOT NULL CHECK(selected IN (0,1)),
                        fraction REAL NOT NULL,
                        seed INTEGER NOT NULL,
                        replay_version TEXT NOT NULL,
                        FOREIGN KEY(question_id) REFERENCES questions(question_id)
                    );

                    CREATE TABLE IF NOT EXISTS replay_steps (
                        model_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        sentence_index INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        score INTEGER,
                        parse_mode TEXT,
                        prompt_sha256 TEXT NOT NULL,
                        error TEXT,
                        PRIMARY KEY(model_id, question_id, sentence_index)
                    );

                    CREATE INDEX IF NOT EXISTS idx_cells_live
                        ON cells(live_status, model_id);
                    CREATE INDEX IF NOT EXISTS idx_cells_post
                        ON cells(post_status, model_id);
                    CREATE INDEX IF NOT EXISTS idx_scores_condition
                        ON scores(condition, model_id, question_id);
                    CREATE INDEX IF NOT EXISTS idx_calls_lookup
                        ON api_calls(condition, model_id, question_id);
                    """
                )

    def write_meta(self, key: str, value: Any) -> None:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
            if row is not None and row["value"] != text:
                raise ValueError(
                    f"existing experiment DB has different {key}; "
                    "use a fresh --out directory"
                )
            con.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)",
                (key, text),
            )

    def register(
        self,
        questions: Sequence[Question],
        models: Mapping[str, str],
    ) -> None:
        with self.connect() as con:
            for q in questions:
                metadata_json = json.dumps(
                    q.metadata, ensure_ascii=False, sort_keys=True
                )
                existing = con.execute(
                    "SELECT family, task, metadata_json FROM questions "
                    "WHERE question_id=?",
                    (q.question_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["family"] != q.family
                        or existing["task"] != q.task
                        or existing["metadata_json"] != metadata_json
                    ):
                        raise ValueError(
                            f"question {q.question_id} differs from existing DB; "
                            "use a fresh --out directory"
                        )
                else:
                    con.execute(
                        "INSERT INTO questions VALUES (?,?,?,?)",
                        (q.question_id, q.family, q.task, metadata_json),
                    )

                for model_id, route in models.items():
                    existing_cell = con.execute(
                        "SELECT model_route FROM cells "
                        "WHERE model_id=? AND question_id=?",
                        (model_id, q.question_id),
                    ).fetchone()
                    if existing_cell is not None:
                        if existing_cell["model_route"] != route:
                            raise ValueError(
                                f"route changed for {model_id}; use a fresh --out"
                            )
                    else:
                        con.execute(
                            "INSERT INTO cells(model_id,model_route,question_id) "
                            "VALUES (?,?,?)",
                            (model_id, route, q.question_id),
                        )

    def status(self, model_id: str, qid: str, condition: str) -> str:
        if condition not in {"live", "post"}:
            raise ValueError(condition)
        with self.connect() as con:
            row = con.execute(
                f"SELECT {condition}_status AS status FROM cells "
                "WHERE model_id=? AND question_id=?",
                (model_id, qid),
            ).fetchone()
        if row is None:
            raise KeyError((model_id, qid))
        return str(row["status"])

    def live_template(self, model_id: str, qid: str) -> str:
        with self.connect() as con:
            row = con.execute(
                "SELECT live_template FROM cells WHERE model_id=? "
                "AND question_id=? AND live_status='complete'",
                (model_id, qid),
            ).fetchone()
        if row is None or row["live_template"] is None:
            raise KeyError((model_id, qid))
        return str(row["live_template"])

    def record_call(
        self,
        *,
        model_id: str,
        qid: str,
        condition: str,
        sentence_index: int | None,
        prompt_hash: str,
        result: ApiResult | None,
        error: str | None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO api_calls(
                    model_id,question_id,condition,sentence_index,prompt_sha256,
                    status,provider,latency_s,prompt_tokens,completion_tokens,
                    response_text,response_json,error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_id,
                    qid,
                    condition,
                    sentence_index,
                    prompt_hash,
                    ("complete" if error is None else ("parse_error" if result is not None else "error")),
                    result.provider if result else None,
                    result.latency_s if result else None,
                    result.prompt_tokens if result else None,
                    result.completion_tokens if result else None,
                    result.text if result else None,
                    json.dumps(
                        result.raw_json, ensure_ascii=False, sort_keys=True
                    )
                    if result
                    else None,
                    error,
                ),
            )

    def save_live(
        self,
        model_id: str,
        qid: str,
        parsed: LiveParse,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE cells SET live_status='complete', live_text=?,
                    live_template=?, live_error=NULL
                WHERE model_id=? AND question_id=?
                """,
                (parsed.raw_text, parsed.template, model_id, qid),
            )
            con.execute(
                "DELETE FROM scores WHERE model_id=? AND question_id=? "
                "AND condition='live'",
                (model_id, qid),
            )
            con.executemany(
                "INSERT INTO scores VALUES (?,?,?,?,?)",
                [
                    (model_id, qid, "live", i, int(score))
                    for i, score in enumerate(parsed.scores)
                ],
            )

    def save_condition_error(
        self,
        model_id: str,
        qid: str,
        condition: str,
        error: str,
    ) -> None:
        if condition not in {"live", "post"}:
            raise ValueError(condition)
        with self.connect() as con:
            con.execute(
                f"UPDATE cells SET {condition}_status='error', "
                f"{condition}_error=? WHERE model_id=? AND question_id=?",
                (error, model_id, qid),
            )

    def save_post(
        self,
        model_id: str,
        qid: str,
        text: str,
        scores: Sequence[int],
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE cells SET post_status='complete', post_text=?,
                    post_error=NULL
                WHERE model_id=? AND question_id=?
                """,
                (text, model_id, qid),
            )
            con.execute(
                "DELETE FROM scores WHERE model_id=? AND question_id=? "
                "AND condition='post'",
                (model_id, qid),
            )
            con.executemany(
                "INSERT INTO scores VALUES (?,?,?,?,?)",
                [
                    (model_id, qid, "post", i, int(score))
                    for i, score in enumerate(scores)
                ],
            )

    def persist_replay_selection(
        self,
        qids: Sequence[str],
        selected: set[str],
        fraction: float,
        seed: int,
    ) -> None:
        with self.connect() as con:
            existing = con.execute(
                "SELECT question_id,selected,fraction,seed,replay_version "
                "FROM replay_selection"
            ).fetchall()

            if existing:
                old_selected = {
                    str(row["question_id"])
                    for row in existing
                    if int(row["selected"]) == 1
                }
                if old_selected != selected:
                    raise ValueError(
                        "replay selection differs from existing DB; use a fresh "
                        "--out directory rather than changing the sample"
                    )
                return

            con.executemany(
                "INSERT INTO replay_selection VALUES (?,?,?,?,?)",
                [
                    (
                        qid,
                        1 if qid in selected else 0,
                        fraction,
                        seed,
                        REPLAY_VERSION,
                    )
                    for qid in qids
                ],
            )

    def completed_replay_scores(self, model_id: str, qid: str) -> list[int]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT sentence_index,score FROM replay_steps
                WHERE model_id=? AND question_id=? AND status='complete'
                ORDER BY sentence_index
                """,
                (model_id, qid),
            ).fetchall()
        result: list[int] = []
        for expected_index, row in enumerate(rows):
            actual = int(row["sentence_index"])
            if actual != expected_index:
                raise RuntimeError(
                    f"replay for {(model_id, qid)} is non-contiguous: "
                    f"expected sentence {expected_index}, found {actual}"
                )
            result.append(int(row["score"]))
        return result

    def save_replay_success(
        self,
        *,
        model_id: str,
        qid: str,
        sentence_index: int,
        score: int,
        parse_mode: str,
        prompt_hash: str,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO replay_steps(
                    model_id,question_id,sentence_index,status,score,parse_mode,
                    prompt_sha256,error
                ) VALUES (?,?,?,'complete',?,?,?,NULL)
                ON CONFLICT(model_id,question_id,sentence_index) DO UPDATE SET
                    status='complete',score=excluded.score,
                    parse_mode=excluded.parse_mode,
                    prompt_sha256=excluded.prompt_sha256,error=NULL
                """,
                (
                    model_id,
                    qid,
                    sentence_index,
                    int(score),
                    parse_mode,
                    prompt_hash,
                ),
            )
            con.execute(
                """
                INSERT INTO scores(
                    model_id,question_id,condition,sentence_index,score
                ) VALUES (?,?, 'replay', ?, ?)
                ON CONFLICT(
                    model_id,question_id,condition,sentence_index
                ) DO UPDATE SET score=excluded.score
                """,
                (model_id, qid, sentence_index, int(score)),
            )

    def save_replay_error(
        self,
        *,
        model_id: str,
        qid: str,
        sentence_index: int,
        prompt_hash: str,
        error: str,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO replay_steps(
                    model_id,question_id,sentence_index,status,score,parse_mode,
                    prompt_sha256,error
                ) VALUES (?,?,?,'error',NULL,NULL,?,?)
                ON CONFLICT(model_id,question_id,sentence_index) DO UPDATE SET
                    status='error',score=NULL,parse_mode=NULL,
                    prompt_sha256=excluded.prompt_sha256,error=excluded.error
                """,
                (model_id, qid, sentence_index, prompt_hash, error),
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.done = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def tick(self, model_id: str, qid: str, ok: bool) -> None:
        with self.lock:
            self.done += 1
            if not ok:
                self.errors += 1
            elapsed = max(0.001, time.monotonic() - self.started)
            rate = self.done / elapsed * 60
            marker = "ok" if ok else "ERROR"
            print(
                f"[{self.label}] {self.done}/{self.total} {marker} "
                f"{model_id} {qid} | errors={self.errors} "
                f"rate={rate:.1f}/min",
                flush=True,
            )


class Runner:
    def __init__(
        self,
        *,
        store: Store,
        client: OpenRouterClient,
        questions: Sequence[Question],
        models: Mapping[str, str],
        workers: int,
        per_model_workers: int,
        live_max_tokens: int,
        post_max_tokens: int,
        replay_max_tokens: int,
    ):
        self.store = store
        self.client = client
        self.questions = list(questions)
        self.models = dict(models)
        self.workers = workers
        self.per_model_workers = per_model_workers
        self.model_semaphores = {
            model_id: threading.Semaphore(per_model_workers)
            for model_id in self.models
        }
        self.model_call_clock_lock = threading.Lock()
        self.model_last_call_started: dict[str, float] = {}
        self.live_max_tokens = live_max_tokens
        self.post_max_tokens = post_max_tokens
        self.replay_max_tokens = replay_max_tokens
        self.qmap = {q.question_id: q for q in questions}

    @staticmethod
    def _prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
        canonical = json.dumps(
            list(messages), ensure_ascii=False, sort_keys=True
        )
        return sha256_text(canonical)

    def _call_model(
        self,
        *,
        model_id: str,
        model_route: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> ApiResult:
        # Keep useful cross-model parallelism without hammering one route.
        with self.model_semaphores[model_id]:
            min_interval = MODEL_MIN_INTERVAL_SECONDS.get(model_id, 0.0)
            if min_interval > 0:
                with self.model_call_clock_lock:
                    previous = self.model_last_call_started.get(model_id)
                if previous is not None:
                    remaining = min_interval - (time.monotonic() - previous)
                    if remaining > 0:
                        time.sleep(remaining)
                with self.model_call_clock_lock:
                    self.model_last_call_started[model_id] = time.monotonic()
            return self.client.call(
                model_id=model_id,
                model_route=model_route,
                messages=messages,
                max_tokens=max_tokens,
            )

    def _run_parallel(self, jobs, label: str) -> None:
        if not jobs:
            print(f"[{label}] nothing pending", flush=True)
            return
        progress = Progress(label, len(jobs))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers
        ) as pool:
            future_map = {
                pool.submit(fn): (model_id, qid)
                for model_id, qid, fn in jobs
            }
            for future in concurrent.futures.as_completed(future_map):
                model_id, qid = future_map[future]
                ok = True
                try:
                    future.result()
                except Exception as exc:  # cell failure is persisted, run continues
                    ok = False
                    print(
                        f"[{label}] failure {model_id} {qid}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                progress.tick(model_id, qid, ok)

    def run_live(self) -> None:
        jobs = []
        # Question-major ordering spreads the first wave across model routes.
        for q in self.questions:
            for model_id, route in self.models.items():
                if self.store.status(model_id, q.question_id, "live") == "complete":
                    continue

                def job(model_id=model_id, route=route, q=q):
                    messages = live_messages(q.task)
                    prompt_hash = self._prompt_hash(messages)
                    result = None
                    try:
                        result = self._call_model(
                            model_id=model_id,
                            model_route=route,
                            messages=messages,
                            max_tokens=self.live_max_tokens,
                        )
                        parsed = parse_live(
                            result.text,
                            expected_min_scores=CASE_SENTENCE_MIN,
                            expected_max_scores=CASE_SENTENCE_MAX,
                        )
                        self.store.save_live(model_id, q.question_id, parsed)
                        self.store.record_call(
                            model_id=model_id,
                            qid=q.question_id,
                            condition="live",
                            sentence_index=None,
                            prompt_hash=prompt_hash,
                            result=result,
                            error=None,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        self.store.save_condition_error(
                            model_id, q.question_id, "live", error
                        )
                        self.store.record_call(
                            model_id=model_id,
                            qid=q.question_id,
                            condition="live",
                            sentence_index=None,
                            prompt_hash=prompt_hash,
                            result=result,
                            error=error,
                        )
                        raise

                jobs.append((model_id, q.question_id, job))
        self._run_parallel(jobs, "live")

    def run_post(self) -> None:
        jobs = []
        for q in self.questions:
            for model_id, route in self.models.items():
                if self.store.status(model_id, q.question_id, "live") != "complete":
                    continue
                if self.store.status(model_id, q.question_id, "post") == "complete":
                    continue

                def job(model_id=model_id, route=route, q=q):
                    template = self.store.live_template(model_id, q.question_id)
                    messages = post_messages(q.task, template)
                    prompt_hash = self._prompt_hash(messages)
                    result = None
                    try:
                        result = self._call_model(
                            model_id=model_id,
                            model_route=route,
                            messages=messages,
                            max_tokens=self.post_max_tokens,
                        )
                        scores = parse_post(result.text, template)
                        self.store.save_post(
                            model_id, q.question_id, result.text, scores
                        )
                        self.store.record_call(
                            model_id=model_id,
                            qid=q.question_id,
                            condition="post",
                            sentence_index=None,
                            prompt_hash=prompt_hash,
                            result=result,
                            error=None,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        self.store.save_condition_error(
                            model_id, q.question_id, "post", error
                        )
                        self.store.record_call(
                            model_id=model_id,
                            qid=q.question_id,
                            condition="post",
                            sentence_index=None,
                            prompt_hash=prompt_hash,
                            result=result,
                            error=error,
                        )
                        raise

                jobs.append((model_id, q.question_id, job))
        self._run_parallel(jobs, "post")

    def run_replay(self, fraction: float, seed: int) -> set[str]:
        qids = [q.question_id for q in self.questions]
        selected = set(select_replay_questions(qids, fraction, seed))
        self.store.persist_replay_selection(
            qids, selected, fraction, seed
        )

        jobs = []
        for qid in sorted(selected):
            q = self.qmap[qid]
            for model_id, route in self.models.items():
                if self.store.status(model_id, qid, "live") != "complete":
                    continue
                # Replay is the sampled third member of a matched triplet: only
                # cells with BOTH live and full post-hoc assessments are eligible.
                if self.store.status(model_id, qid, "post") != "complete":
                    continue

                def job(model_id=model_id, route=route, q=q):
                    template = self.store.live_template(model_id, q.question_id)
                    n_sentences = len(split_template(template)) - 1

                    # Critical invariant: resume ONLY from replay_steps. Neither
                    # live nor post score values are read here.
                    replay_scores = self.store.completed_replay_scores(
                        model_id, q.question_id
                    )
                    if len(replay_scores) > n_sentences:
                        raise RuntimeError(
                            "more replay scores than live score slots"
                        )

                    for sentence_index in range(
                        len(replay_scores), n_sentences
                    ):
                        prefix = build_replay_prefix(
                            template, replay_scores, sentence_index
                        )
                        messages = replay_messages(q.task, prefix)
                        prompt_hash = self._prompt_hash(messages)
                        result = None
                        try:
                            result = self._call_model(
                                model_id=model_id,
                                model_route=route,
                                messages=messages,
                                max_tokens=self.replay_max_tokens,
                            )
                            score, parse_mode = parse_single_replay_score(
                                result.text, replay_scores
                            )
                            self.store.save_replay_success(
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                score=score,
                                parse_mode=parse_mode,
                                prompt_hash=prompt_hash,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=None,
                            )
                            replay_scores.append(score)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            self.store.save_replay_error(
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                error=error,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=error,
                            )
                            # Important: later replay sentences must not run after
                            # a missing earlier replay score.
                            raise

                jobs.append((model_id, qid, job))

        self._run_parallel(jobs, "replay")
        return selected


# ---------------------------------------------------------------------------
# Exports / analysis
# ---------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def export_results(out_dir: Path, store: Store) -> dict[str, Any]:
    with store.connect() as con:
        score_rows = con.execute(
            """
            SELECT s.model_id,s.question_id,q.family,s.condition,
                   s.sentence_index,s.score
            FROM scores s
            JOIN questions q USING(question_id)
            ORDER BY s.model_id,s.question_id,s.sentence_index,s.condition
            """
        ).fetchall()

        paired_rows = con.execute(
            """
            SELECT
                l.model_id,l.question_id,q.family,l.sentence_index,
                l.score AS live_score,
                p.score AS post_score,
                r.score AS replay_score
            FROM scores l
            JOIN questions q USING(question_id)
            LEFT JOIN scores p
              ON p.model_id=l.model_id
             AND p.question_id=l.question_id
             AND p.sentence_index=l.sentence_index
             AND p.condition='post'
            LEFT JOIN scores r
              ON r.model_id=l.model_id
             AND r.question_id=l.question_id
             AND r.sentence_index=l.sentence_index
             AND r.condition='replay'
            WHERE l.condition='live'
            ORDER BY l.model_id,l.question_id,l.sentence_index
            """
        ).fetchall()

        cell_rows = con.execute(
            """
            SELECT model_id,question_id,live_status,live_error,
                   post_status,post_error
            FROM cells ORDER BY model_id,question_id
            """
        ).fetchall()

        replay_selected = [
            row["question_id"]
            for row in con.execute(
                "SELECT question_id FROM replay_selection "
                "WHERE selected=1 ORDER BY question_id"
            ).fetchall()
        ]

    _write_csv(
        out_dir / "scores_long.csv",
        ["model_id","question_id","family","condition","sentence_index","score"],
        [
            [
                r["model_id"], r["question_id"], r["family"], r["condition"],
                r["sentence_index"], r["score"],
            ]
            for r in score_rows
        ],
    )
    _write_csv(
        out_dir / "paired_scores.csv",
        [
            "model_id","question_id","family","sentence_index",
            "live_score","post_score","replay_score",
        ],
        [
            [
                r["model_id"], r["question_id"], r["family"],
                r["sentence_index"], r["live_score"], r["post_score"],
                r["replay_score"],
            ]
            for r in paired_rows
        ],
    )
    _write_csv(
        out_dir / "cell_status.csv",
        [
            "model_id","question_id","live_status","live_error",
            "post_status","post_error",
        ],
        [
            [
                r["model_id"], r["question_id"], r["live_status"],
                r["live_error"], r["post_status"], r["post_error"],
            ]
            for r in cell_rows
        ],
    )

    by_model: dict[str, list[sqlite3.Row]] = {}
    for row in paired_rows:
        by_model.setdefault(str(row["model_id"]), []).append(row)

    summary_rows = []
    summary_json: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "replay_version": REPLAY_VERSION,
        "replay_question_ids": replay_selected,
        "models": [],
    }

    for model_id, rows in sorted(by_model.items()):
        lp = [
            (float(r["live_score"]), float(r["post_score"]))
            for r in rows
            if r["post_score"] is not None
        ]
        lr = [
            (float(r["live_score"]), float(r["replay_score"]))
            for r in rows
            if r["replay_score"] is not None
        ]
        pr = [
            (float(r["post_score"]), float(r["replay_score"]))
            for r in rows
            if r["post_score"] is not None and r["replay_score"] is not None
        ]

        metrics = {
            "model_id": model_id,
            **_pair_metrics("live_post", lp),
            **_pair_metrics("live_replay", lr),
            **_pair_metrics("post_replay", pr),
        }
        summary_rows.append(metrics)
        summary_json["models"].append(metrics)

    fields = [
        "model_id",
        "live_post_n","live_post_bias_first_minus_second",
        "live_post_mae","live_post_pearson","live_post_spearman",
        "live_replay_n","live_replay_bias_first_minus_second",
        "live_replay_mae","live_replay_pearson","live_replay_spearman",
        "post_replay_n","post_replay_bias_first_minus_second",
        "post_replay_mae","post_replay_pearson","post_replay_spearman",
    ]
    _write_csv(
        out_dir / "model_summary.csv",
        fields,
        [[row.get(field) for field in fields] for row in summary_rows],
    )

    status_counts: dict[str, int] = {}
    for row in cell_rows:
        for condition in ("live", "post"):
            key = f"{condition}_{row[f'{condition}_status']}"
            status_counts[key] = status_counts.get(key, 0) + 1

    summary_json["cell_status_counts"] = status_counts
    summary_json["score_rows"] = len(score_rows)
    summary_json["paired_live_rows"] = len(paired_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_json


def _pair_metrics(
    prefix: str, pairs: Sequence[tuple[float, float]]
) -> dict[str, Any]:
    if not pairs:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_bias_first_minus_second": None,
            f"{prefix}_mae": None,
            f"{prefix}_pearson": None,
            f"{prefix}_spearman": None,
        }
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    diffs = [a - b for a, b in pairs]
    return {
        f"{prefix}_n": len(pairs),
        f"{prefix}_bias_first_minus_second": statistics.fmean(diffs),
        f"{prefix}_mae": statistics.fmean(abs(v) for v in diffs),
        f"{prefix}_pearson": pearson(xs, ys),
        f"{prefix}_spearman": spearman(xs, ys),
    }


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    questions: Sequence[Question],
    models: Mapping[str, str],
    replay_ids: Sequence[str],
) -> Path:
    script_path = Path(__file__).resolve()
    manifest = {
        "code_version": CODE_VERSION,
        "script_filename": script_path.name,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "replay_version": REPLAY_VERSION,
        "seed": args.seed,
        "cases": len(questions),
        "replay_fraction": args.replay_fraction,
        "temperature": args.temperature,
        "workers": args.workers,
        "per_model_workers": args.per_model_workers,
        "live_max_tokens": args.live_max_tokens,
        "post_max_tokens": args.post_max_tokens,
        "replay_max_tokens": args.replay_max_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "ai_score_instruction": AI_SCORE_INSTRUCTION,
        "models": models,
        "model_provider_preferences": MODEL_PROVIDER_PREFERENCES,
        "model_min_interval_seconds": MODEL_MIN_INTERVAL_SECONDS,
        "replay_question_ids": list(replay_ids),
        "questions": [
            {
                "question_id": q.question_id,
                "family": q.family,
                "task": q.task,
                "metadata": q.metadata,
            }
            for q in questions
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def model_selection(spec: str) -> dict[str, str]:
    if spec.strip().lower() == "all":
        return dict(MODEL_SPECS)
    aliases = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [a for a in aliases if a not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            f"unknown model aliases: {', '.join(unknown)}. "
            f"Known: {', '.join(MODEL_SPECS)}"
        )
    return {alias: MODEL_SPECS[alias] for alias in aliases}


def self_test() -> None:
    task = "Write 6-8 sentences about a small everyday event."
    common = common_user_prefix(task)
    assert common in live_messages(task)[1]["content"]

    live = "First sentence. <AI SCORE: 12>\nSecond sentence! <AI SCORE: 87>"
    parsed = parse_live(live)
    assert parsed.scores == (12, 87)
    assert parsed.template == (
        "First sentence. <AI SCORE: ?>\nSecond sentence! <AI SCORE: ?>"
    )
    assert common in post_messages(task, parsed.template)[1]["content"]

    llama_style = (
        "First sentence <AI SCORE: 12>. "
        "Second sentence <AI SCORE: 87>."
    )
    parsed_llama = parse_live(llama_style)
    assert parsed_llama.scores == (12, 87)
    assert parsed_llama.template == (
        "First sentence. <AI SCORE: ?> "
        "Second sentence. <AI SCORE: ?>"
    )

    try:
        parse_live(
            "First. <AI SCORE: 10> Second. <AI SCORE: 20>",
            expected_min_scores=6,
            expected_max_scores=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("too few live score tags must fail")

    try:
        parse_live("First. <AI SCORE: 10> unfinished continuation")
    except ValueError:
        pass
    else:
        raise AssertionError("unfinished post-score prose must fail")

    try:
        parse_live("First. Second.\n<AI SCORE: 10>\n<AI SCORE: 20>")
    except ValueError:
        pass
    else:
        raise AssertionError("detached score block must fail")

    post = "<AI SCORE: 22>\n<AI SCORE: 70>"
    assert parse_post(post, parsed.template) == (22, 70)
    assert parse_post(
        "AI SCORE: 22\nAI SCORE: 70", parsed.template
    ) == (22, 70)
    assert parse_post(
        "<AI SCORE: (22)>\n<AI SCORE: (70)>", parsed.template
    ) == (22, 70)
    assert parse_post("22\n70", parsed.template) == (22, 70)

    # Malformed live score-like tags must never silently shift alignment.
    for malformed in (
        "First sentence. <AI SCORE: -1>\nSecond sentence. <AI SCORE: 20>",
        "First sentence. <AI SCORE: III>\nSecond sentence. <AI SCORE: 20>",
    ):
        try:
            parse_live(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed live score-like tag must fail")
    try:
        parse_post("<AI SCORE: 22>", parsed.template)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong POST score count must fail")

    p0 = build_replay_prefix(parsed.template, [], 0)
    assert p0 == "First sentence. <AI SCORE: ?>"
    assert "Second sentence" not in p0
    assert common in replay_messages(task, p0)[1]["content"]

    p1 = build_replay_prefix(parsed.template, [41], 1)
    assert p1 == (
        "First sentence. <AI SCORE: 41>\nSecond sentence! <AI SCORE: ?>"
    )
    assert "12" not in p1 and "87" not in p1 and "22" not in p1

    assert parse_single_replay_score(
        "<AI SCORE: 54>", []
    ) == (54, "exact_tag")
    assert parse_single_replay_score(
        "My answer: <AI SCORE: 54>", []
    ) == (54, "tag_with_extra_text")
    assert parse_single_replay_score(
        "AI SCORE: 54", []
    ) == (54, "bare_score")
    assert parse_single_replay_score(
        "First sentence. <AI SCORE: 41>\n"
        "Second sentence! <AI SCORE: 57>",
        [41],
    ) == (57, "repeated_prefix")

    try:
        parse_single_replay_score(
            "<AI SCORE: 10> or <AI SCORE: 20>", []
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous multiple replay scores must fail")

    qids = [f"q{i:03d}" for i in range(400)]
    selected = select_replay_questions(qids, 0.10, 20260903)
    assert len(selected) == 40
    assert selected == select_replay_questions(
        reversed(qids), 0.10, 20260903
    )

    questions = generate_questions(20, 20260903)
    assert len(questions) == 20
    counts = {family: 0 for family in FAMILY_NAMES}
    for q in questions:
        counts[q.family] += 1
    assert set(counts.values()) == {2}

    print("self-test: all checks passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V5 AI-likeness live/post/replay self-assessment experiment"
        )
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("self_confidence_v5"))
    parser.add_argument("--replay-fraction", type=float, default=0.10)
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "all, or comma-separated aliases such as "
            "gpt-5.5,deepseek-v3.2"
        ),
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--per-model-workers",
        type=int,
        default=1,
        help="maximum simultaneous API calls to one model route",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--live-max-tokens", type=int, default=4096)
    parser.add_argument("--post-max-tokens", type=int, default=4096)
    parser.add_argument("--replay-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--conditions",
        default="live,post,replay",
        help="comma-separated subset in execution order; default all three",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show exact plan and write manifest, but make no API calls",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run local invariants and exit; no API key/network needed",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    print(
        f"Self-confidence runner V{CODE_VERSION} | "
        f"prompt={PROMPT_VERSION} | schema={SCHEMA_VERSION}",
        flush=True,
    )
    print(f"Script: {script_path.name} | SHA256={script_sha}", flush=True)

    if args.self_test:
        self_test()
        return 0

    if not 0 <= args.replay_fraction <= 1:
        parser.error("--replay-fraction must be between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.per_model_workers < 1:
        parser.error("--per-model-workers must be >= 1")

    try:
        models = model_selection(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    conditions = [
        item.strip() for item in args.conditions.split(",") if item.strip()
    ]
    allowed = {"live", "post", "replay"}
    if not conditions or any(item not in allowed for item in conditions):
        parser.error("--conditions must contain live, post, and/or replay")
    order = {"live": 0, "post": 1, "replay": 2}
    if conditions != sorted(conditions, key=order.get):
        parser.error(
            "--conditions must be in causal order: live, then post, then replay"
        )

    questions = generate_questions(args.cases, args.seed)
    replay_ids = select_replay_questions(
        [q.question_id for q in questions],
        args.replay_fraction,
        args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(
        args.out,
        args=args,
        questions=questions,
        models=models,
        replay_ids=replay_ids,
    )

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Questions: {len(questions)}", flush=True)
    print(f"Models: {len(models)}", flush=True)
    print(
        f"Concurrency: workers={args.workers}, "
        f"per-model-workers={args.per_model_workers}",
        flush=True,
    )
    print(
        f"Token caps: live={args.live_max_tokens}, "
        f"post={args.post_max_tokens}, replay={args.replay_max_tokens}",
        flush=True,
    )
    if MODEL_PROVIDER_PREFERENCES:
        print(
            "Provider pins: " + json.dumps(MODEL_PROVIDER_PREFERENCES, sort_keys=True),
            flush=True,
        )
    if MODEL_MIN_INTERVAL_SECONDS:
        print(
            "Model min intervals: "
            + json.dumps(MODEL_MIN_INTERVAL_SECONDS, sort_keys=True),
            flush=True,
        )
    print(f"Live cells: {len(questions) * len(models)}", flush=True)
    print(f"Post cells: {len(questions) * len(models)}", flush=True)
    print(
        f"Replay questions: {len(replay_ids)} "
        f"({args.replay_fraction:.1%} requested)",
        flush=True,
    )
    print(
        f"Replay model/question cells: {len(replay_ids) * len(models)}",
        flush=True,
    )
    print(
        "Replay policy: selected questions only; requires live + post; full "
        "sequential replay sentence 1 -> N; prior scores come only from the "
        "same replay",
        flush=True,
    )
    print("Models/routes:", flush=True)
    for alias, route in models.items():
        print(f"  {alias} -> {route}", flush=True)
    print("Replay question IDs:", ", ".join(replay_ids) or "(none)", flush=True)

    if args.dry_run:
        print("DRY RUN: no API calls made; SQLite DB not created.", flush=True)
        return 0

    # Match common local usage: check current .env, then script-directory .env.
    load_simple_dotenv(Path.cwd() / ".env")
    load_simple_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        parser.error(
            "OPENROUTER_API_KEY is not set. Export it in Git Bash or put it "
            "in .env as OPENROUTER_API_KEY=..."
        )

    db_path = args.out / "experiment.sqlite3"
    store = Store(db_path)
    store.write_meta("code_version", CODE_VERSION)
    store.write_meta("script_sha256", script_sha)
    store.write_meta("schema_version", SCHEMA_VERSION)
    store.write_meta("prompt_version", PROMPT_VERSION)
    store.write_meta("replay_version", REPLAY_VERSION)
    store.write_meta("seed", args.seed)
    store.write_meta("cases", len(questions))
    store.write_meta("replay_fraction", args.replay_fraction)
    store.write_meta("temperature", args.temperature)
    store.write_meta("workers", args.workers)
    store.write_meta("per_model_workers", args.per_model_workers)
    store.write_meta("live_max_tokens", args.live_max_tokens)
    store.write_meta("post_max_tokens", args.post_max_tokens)
    store.write_meta("replay_max_tokens", args.replay_max_tokens)
    store.write_meta("models", models)
    store.write_meta("model_provider_preferences", MODEL_PROVIDER_PREFERENCES)
    store.write_meta("model_min_interval_seconds", MODEL_MIN_INTERVAL_SECONDS)
    store.write_meta("system_prompt", SYSTEM_PROMPT)
    store.write_meta("ai_score_instruction", AI_SCORE_INSTRUCTION)
    store.register(questions, models)

    client = OpenRouterClient(
        api_key=api_key,
        temperature=args.temperature,
        timeout_s=args.timeout,
        retries=args.retries,
    )
    runner = Runner(
        store=store,
        client=client,
        questions=questions,
        models=models,
        workers=args.workers,
        per_model_workers=args.per_model_workers,
        live_max_tokens=args.live_max_tokens,
        post_max_tokens=args.post_max_tokens,
        replay_max_tokens=args.replay_max_tokens,
    )

    if "live" in conditions:
        runner.run_live()
        export_results(args.out, store)

    if "post" in conditions:
        runner.run_post()
        export_results(args.out, store)

    if "replay" in conditions:
        runner.run_replay(args.replay_fraction, args.seed)
        export_results(args.out, store)

    summary = export_results(args.out, store)
    print(f"SQLite: {db_path}", flush=True)
    print(f"Scores: {args.out / 'scores_long.csv'}", flush=True)
    print(f"Pairs: {args.out / 'paired_scores.csv'}", flush=True)
    print(f"Model summary: {args.out / 'model_summary.csv'}", flush=True)
    print(f"Summary: {args.out / 'summary.json'}", flush=True)
    print(
        "Cell status counts: "
        + json.dumps(summary.get("cell_status_counts", {}), sort_keys=True),
        flush=True,
    )
    return 0




# ---------------------------------------------------------------------------
# V5.5 adaptive/sequential experiment layer
# ---------------------------------------------------------------------------

ADAPTIVE_VERSION = "fixed-looks-bonferroni-promptbank-v2"
DEFAULT_PRIMARY_MODEL = "gpt-5.5"
DEFAULT_STATE_MODEL = "gemini-3.1-pro-preview"


class NonRetryableModelError(RuntimeError):
    """A valid provider response that should not be retried automatically."""


class FastFailOpenRouterClient:
    """
    OpenRouter client with bounded transient retries.

    V5.4 could spend several minutes retrying content_filter and persistent 429
    responses. V5.5 retries only genuinely transient transport/provider errors,
    caps backoff, never retries content_filter, and makes at most one immediate
    larger-token retry for an empty finish_reason='length' completion.
    """

    RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        api_key: str,
        temperature: float,
        timeout_s: float,
        retries: int,
        max_retry_delay_s: float,
    ):
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = max(1, retries)
        self.max_retry_delay_s = max(0.0, max_retry_delay_s)

    @staticmethod
    def _extract_text(data: Mapping[str, Any]) -> tuple[str | None, str | None]:
        choices = data.get("choices") or []
        if not choices:
            raise NonRetryableModelError("OpenRouter response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "".join(parts)
        return (content if isinstance(content, str) else None, choice.get("finish_reason"))

    def _one_request(
        self,
        *,
        model_route: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> tuple[Mapping[str, Any], float]:
        payload = {
            "model": model_route,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            # Deliberately allow OpenRouter provider fallback in V5.5. The
            # provider actually used is retained in api_calls for auditability.
            "provider": {"allow_fallbacks": True},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://example.invalid/anonymous-review",
                "X-Title": "Self Confidence V5 Adaptive",
            },
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw_bytes = response.read()
        latency = time.perf_counter() - started
        return json.loads(raw_bytes.decode("utf-8")), latency

    def call(
        self,
        *,
        model_id: str,
        model_route: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> ApiResult:
        del model_id  # provider pinning is intentionally disabled in V5.5
        last_error: Exception | None = None
        token_budget = max_tokens
        used_length_retry = False

        for attempt in range(1, self.retries + 1):
            try:
                data, latency = self._one_request(
                    model_route=model_route,
                    messages=messages,
                    max_tokens=token_budget,
                )
                content, finish_reason = self._extract_text(data)
                if isinstance(content, str) and content.strip():
                    usage = data.get("usage") or {}
                    return ApiResult(
                        text=content,
                        raw_json=data,
                        provider=data.get("provider"),
                        latency_s=latency,
                        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                        completion_tokens=_int_or_none(usage.get("completion_tokens")),
                    )

                if finish_reason == "length" and not used_length_retry:
                    # One immediate retry with more room; no exponential sleep.
                    used_length_retry = True
                    token_budget = min(max(token_budget * 2, token_budget + 512), 8192)
                    continue

                if finish_reason == "content_filter":
                    raise NonRetryableModelError(
                        "model returned empty final content (finish_reason='content_filter')"
                    )
                raise NonRetryableModelError(
                    f"model returned empty final content (finish_reason={finish_reason!r})"
                )

            except NonRetryableModelError:
                raise
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in self.RETRYABLE_HTTP or attempt >= self.retries:
                    break
                delay = min(self.max_retry_delay_s, 2.0 * (2 ** (attempt - 1)))
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        delay = min(self.max_retry_delay_s, max(delay, float(retry_after)))
                    except ValueError:
                        pass
                if delay > 0:
                    time.sleep(delay + random.random() * 0.4)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = min(self.max_retry_delay_s, 2.0 * (2 ** (attempt - 1)))
                if delay > 0:
                    time.sleep(delay + random.random() * 0.4)

        assert last_error is not None
        raise last_error


def _adaptive_schema(store: Store) -> None:
    with store.connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS adaptive_looks (
                look_cases INTEGER PRIMARY KEY,
                look_index INTEGER NOT NULL,
                alpha_boundary REAL NOT NULL,
                primary_endpoint TEXT NOT NULL,
                primary_n INTEGER,
                primary_effect REAL,
                primary_p REAL,
                state_n INTEGER,
                state_effect REAL,
                state_p REAL,
                modeldep_n INTEGER,
                modeldep_effect REAL,
                modeldep_p REAL,
                decision TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def _persist_replay_selection_incremental(
    store: Store,
    qids: Sequence[str],
    selected: set[str],
    fraction: float,
    seed: int,
) -> None:
    with store.connect() as con:
        for qid in qids:
            desired = 1 if qid in selected else 0
            row = con.execute(
                "SELECT selected,fraction,seed,replay_version FROM replay_selection "
                "WHERE question_id=?",
                (qid,),
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO replay_selection VALUES (?,?,?,?,?)",
                    (qid, desired, fraction, seed, REPLAY_VERSION),
                )
            else:
                if (
                    int(row["selected"]) != desired
                    or abs(float(row["fraction"]) - fraction) > 1e-12
                    or int(row["seed"]) != seed
                    or str(row["replay_version"]) != REPLAY_VERSION
                ):
                    raise ValueError(
                        f"replay selection changed for {qid}; use a fresh --out directory"
                    )


def select_replay_questions_batched(
    questions: Sequence[Question],
    *,
    fraction: float,
    batch_size: int,
    seed: int,
) -> tuple[str, ...]:
    """Deterministically sample replay questions independently within each batch."""
    if not 0 <= fraction <= 1:
        raise ValueError("replay fraction must be in [0,1]")
    ordered = sorted(questions, key=lambda q: int(q.metadata["case_index"]))
    selected: list[str] = []
    for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
        batch = ordered[start:start + batch_size]
        if not batch or fraction == 0:
            continue
        n = max(1, int(round(len(batch) * fraction)))
        n = min(n, len(batch))
        rng = random.Random(seed + 104729 * (batch_index + 1))
        selected.extend(rng.sample([q.question_id for q in batch], n))
    return tuple(sorted(selected))


class AdaptiveRunner(Runner):
    def run_replay_selected(self, selected_qids: set[str]) -> None:
        jobs = []
        for qid in sorted(selected_qids):
            q = self.qmap[qid]
            for model_id, route in self.models.items():
                if self.store.status(model_id, qid, "live") != "complete":
                    continue
                if self.store.status(model_id, qid, "post") != "complete":
                    continue

                def job(model_id=model_id, route=route, q=q):
                    template = self.store.live_template(model_id, q.question_id)
                    n_sentences = len(split_template(template)) - 1
                    replay_scores = self.store.completed_replay_scores(model_id, q.question_id)
                    if len(replay_scores) > n_sentences:
                        raise RuntimeError("more replay scores than live score slots")
                    for sentence_index in range(len(replay_scores), n_sentences):
                        prefix = build_replay_prefix(template, replay_scores, sentence_index)
                        messages = replay_messages(q.task, prefix)
                        prompt_hash = self._prompt_hash(messages)
                        result = None
                        try:
                            result = self._call_model(
                                model_id=model_id,
                                model_route=route,
                                messages=messages,
                                max_tokens=self.replay_max_tokens,
                            )
                            score, parse_mode = parse_single_replay_score(result.text, replay_scores)
                            self.store.save_replay_success(
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                score=score,
                                parse_mode=parse_mode,
                                prompt_hash=prompt_hash,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=None,
                            )
                            replay_scores.append(score)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            self.store.save_replay_error(
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                error=error,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=error,
                            )
                            raise

                jobs.append((model_id, qid, job))
        self._run_parallel(jobs, "replay")


def _question_means(
    store: Store, model_id: str, condition: str,
    question_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    params: list[Any] = [model_id, condition]
    where_extra = ""
    if question_ids is not None:
        qids = list(dict.fromkeys(str(q) for q in question_ids))
        if not qids:
            return {}
        where_extra = " AND question_id IN (" + ",".join("?" for _ in qids) + ")"
        params.extend(qids)
    with store.connect() as con:
        rows = con.execute(
            f"""
            SELECT question_id, AVG(score) AS mean_score
            FROM scores
            WHERE model_id=? AND condition=? {where_extra}
            GROUP BY question_id
            """,
            tuple(params),
        ).fetchall()
    return {str(r["question_id"]): float(r["mean_score"]) for r in rows}


def _paired_condition_means(
    store: Store, model_id: str, a: str, b: str,
    question_ids: Sequence[str] | None = None,
) -> list[tuple[str, float, float]]:
    am = _question_means(store, model_id, a, question_ids)
    bm = _question_means(store, model_id, b, question_ids)
    return [(qid, am[qid], bm[qid]) for qid in sorted(am.keys() & bm.keys())]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clamp_corr(r: float) -> float:
    return max(-0.999999, min(0.999999, r))


def _spearman_greater_p(
    xs: Sequence[float], ys: Sequence[float], rho0: float
) -> tuple[float | None, float | None]:
    """
    Approximate one-sided Fisher-z test for Spearman rho > rho0.
    The independent unit is one question (question-level mean scores).
    """
    if len(xs) != len(ys) or len(xs) < 4:
        return None, None
    r = spearman(xs, ys)
    if r is None:
        return None, None
    r_c = _clamp_corr(float(r))
    r0_c = _clamp_corr(float(rho0))
    z = (math.atanh(r_c) - math.atanh(r0_c)) * math.sqrt(len(xs) - 3)
    return float(r), max(0.0, min(1.0, 1.0 - _normal_cdf(z)))


def _binomial_upper_tail(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)


def _sign_test_greater(
    values: Sequence[float], threshold: float
) -> tuple[int, float, float, float]:
    """Exact one-sided sign test of median(values) > threshold."""
    non_ties = [v for v in values if abs(v - threshold) > 1e-12]
    successes = sum(v > threshold for v in non_ties)
    p = _binomial_upper_tail(successes, len(non_ties)) if non_ties else 1.0
    mean_effect = statistics.fmean(values) if values else float("nan")
    median_effect = statistics.median(values) if values else float("nan")
    return len(non_ties), mean_effect, median_effect, p


def _load_external_question_means(
    path: Path, primary_model: str
) -> dict[str, float]:
    if not path.exists():
        return {}
    values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"model_id", "question_id", "human_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"external ratings CSV missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if row.get("model_id", "").strip() != primary_model:
                continue
            raw = row.get("human_score", "").strip()
            if not raw:
                continue
            score = float(raw)
            if not 0 <= score <= 100:
                raise ValueError(f"external human_score outside 0..100: {score}")
            qid = row["question_id"].strip()
            values.setdefault(qid, []).append(score)
    return {qid: statistics.fmean(vs) for qid, vs in values.items() if vs}


def write_external_ratings_template(
    out_dir: Path, store: Store, primary_model: str,
    question_ids: Sequence[str] | None = None,
) -> Path:
    path = out_dir / "external_ratings_template.csv"
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT c.question_id,c.live_template
            FROM cells c
            WHERE c.model_id=? AND c.live_status='complete'
            ORDER BY c.question_id
            """,
            (primary_model,),
        ).fetchall()
    allowed = None if question_ids is None else set(str(q) for q in question_ids)
    output: list[list[Any]] = []
    for row in rows:
        if allowed is not None and str(row["question_id"]) not in allowed:
            continue
        template = str(row["live_template"])
        pieces = split_template(template)
        for sentence_index, piece in enumerate(pieces[:-1]):
            sentence = piece.strip()
            output.append([primary_model, row["question_id"], sentence_index, sentence, ""])
    _write_csv(
        path,
        ["model_id", "question_id", "sentence_index", "sentence", "human_score"],
        output,
    )
    return path


def _planned_looks(min_cases: int, max_cases: int, batch_size: int) -> list[int]:
    if min_cases <= 0 or max_cases < min_cases or batch_size <= 0:
        raise ValueError("invalid adaptive sample-size settings")
    if min_cases % batch_size != 0 or max_cases % batch_size != 0:
        raise ValueError("--min-cases and --max-cases must be multiples of --batch-size")
    return list(range(min_cases, max_cases + 1, batch_size))


def evaluate_adaptive_look(
    *,
    store: Store,
    cases_attempted: int,
    look_index: int,
    alpha_boundary: float,
    primary_model: str,
    state_model: str,
    primary_endpoint: str,
    external_ratings: Path | None,
    rho0: float,
    min_state_shift: float,
    min_model_difference: float,
    hard_limit: bool,
    question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    # Primary: predeclared primary model. Automatic mode uses LIVE vs POST;
    # external-human mode instead correlates LIVE with supplied human ratings.
    awaiting_ratings = False
    ratings_required = 0
    ratings_available = 0
    if primary_endpoint == "external-human":
        live = _question_means(store, primary_model, "live", question_ids)
        human = _load_external_question_means(external_ratings, primary_model) if external_ratings else {}
        ratings_required = len(live)
        ratings_available = len(live.keys() & human.keys())
        awaiting_ratings = ratings_available < ratings_required
        qids = sorted(live.keys() & human.keys())
        xs = [live[q] for q in qids]
        ys = [human[q] for q in qids]
    else:
        paired = _paired_condition_means(store, primary_model, "live", "post", question_ids)
        qids = [q for q, _, _ in paired]
        xs = [a for _, a, _ in paired]
        ys = [b for _, _, b in paired]
    primary_effect, primary_p = _spearman_greater_p(xs, ys, rho0)

    # Prespecified state distortion: STATE model POST-LIVE exceeds a meaningful
    # shift threshold on a majority of questions (exact one-sided sign test).
    state_pairs = _paired_condition_means(store, state_model, "live", "post", question_ids)
    state_shift_by_q = {q: post - live for q, live, post in state_pairs}
    state_n, state_mean, state_median, state_p = _sign_test_greater(
        list(state_shift_by_q.values()), min_state_shift
    )

    # Prespecified model dependence: the state shift for STATE model exceeds
    # the state shift for PRIMARY model by a meaningful amount on matched qs.
    primary_pairs = _paired_condition_means(store, primary_model, "live", "post", question_ids)
    primary_shift_by_q = {q: post - live for q, live, post in primary_pairs}
    common = sorted(state_shift_by_q.keys() & primary_shift_by_q.keys())
    contrasts = [state_shift_by_q[q] - primary_shift_by_q[q] for q in common]
    modeldep_n, modeldep_mean, modeldep_median, modeldep_p = _sign_test_greater(
        contrasts, min_model_difference
    )

    primary_ok = (
        primary_effect is not None
        and primary_p is not None
        and primary_effect > rho0
        and primary_p <= alpha_boundary
    )
    state_ok = (
        state_n > 0 and state_mean > min_state_shift and state_p <= alpha_boundary
    )
    modeldep_ok = (
        modeldep_n > 0
        and modeldep_mean > min_model_difference
        and modeldep_p <= alpha_boundary
    )
    success = primary_ok and state_ok and modeldep_ok
    if awaiting_ratings:
        decision = "awaiting_ratings"
    else:
        decision = "success" if success else ("hard_limit" if hard_limit else "continue")

    return {
        "adaptive_version": ADAPTIVE_VERSION,
        "cases_attempted": cases_attempted,
        "look_index": look_index,
        "alpha_boundary": alpha_boundary,
        "primary": {
            "endpoint": primary_endpoint,
            "model": primary_model,
            "n_questions": len(xs),
            "spearman": primary_effect,
            "rho0": rho0,
            "p_one_sided": primary_p,
            "ratings_required": ratings_required if primary_endpoint == "external-human" else None,
            "ratings_available": ratings_available if primary_endpoint == "external-human" else None,
            "passed": primary_ok,
        },
        "state_distortion": {
            "model": state_model,
            "n_non_ties": state_n,
            "mean_post_minus_live": state_mean,
            "median_post_minus_live": state_median,
            "minimum_meaningful_shift": min_state_shift,
            "p_sign_one_sided": state_p,
            "passed": state_ok,
        },
        "model_dependence": {
            "contrast": f"({state_model} post-live) - ({primary_model} post-live)",
            "n_non_ties": modeldep_n,
            "mean_contrast": modeldep_mean,
            "median_contrast": modeldep_median,
            "minimum_meaningful_difference": min_model_difference,
            "p_sign_one_sided": modeldep_p,
            "passed": modeldep_ok,
        },
        "decision": decision,
        "confirmatory_scope": (
            "external-validity + state effects"
            if primary_endpoint == "external-human"
            else "internal self-assessment consistency + state effects"
        ),
    }


def record_adaptive_look(store: Store, result: Mapping[str, Any]) -> None:
    p = result["primary"]
    s = result["state_distortion"]
    m = result["model_dependence"]
    with store.connect() as con:
        con.execute(
            """
            INSERT INTO adaptive_looks(
                look_cases,look_index,alpha_boundary,primary_endpoint,
                primary_n,primary_effect,primary_p,state_n,state_effect,state_p,
                modeldep_n,modeldep_effect,modeldep_p,decision,details_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(look_cases) DO UPDATE SET
                look_index=excluded.look_index,
                alpha_boundary=excluded.alpha_boundary,
                primary_endpoint=excluded.primary_endpoint,
                primary_n=excluded.primary_n,
                primary_effect=excluded.primary_effect,
                primary_p=excluded.primary_p,
                state_n=excluded.state_n,
                state_effect=excluded.state_effect,
                state_p=excluded.state_p,
                modeldep_n=excluded.modeldep_n,
                modeldep_effect=excluded.modeldep_effect,
                modeldep_p=excluded.modeldep_p,
                decision=excluded.decision,
                details_json=excluded.details_json,
                created_at=CURRENT_TIMESTAMP
            """,
            (
                result["cases_attempted"],
                result["look_index"],
                result["alpha_boundary"],
                p["endpoint"],
                p["n_questions"],
                p["spearman"],
                p["p_one_sided"],
                s["n_non_ties"],
                s["mean_post_minus_live"],
                s["p_sign_one_sided"],
                m["n_non_ties"],
                m["mean_contrast"],
                m["p_sign_one_sided"],
                result["decision"],
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            ),
        )


def export_adaptive_looks(out_dir: Path, store: Store) -> None:
    with store.connect() as con:
        rows = con.execute(
            "SELECT * FROM adaptive_looks ORDER BY look_cases"
        ).fetchall()
    fields = [
        "look_cases", "look_index", "alpha_boundary", "primary_endpoint",
        "primary_n", "primary_effect", "primary_p", "state_n", "state_effect",
        "state_p", "modeldep_n", "modeldep_effect", "modeldep_p", "decision",
        "created_at",
    ]
    _write_csv(
        out_dir / "adaptive_looks.csv",
        fields,
        [[r[f] for f in fields] for r in rows],
    )
    payload = [json.loads(str(r["details_json"])) for r in rows]
    (out_dir / "adaptive_looks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_adaptive_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    questions: Sequence[Question],
    models: Mapping[str, str],
    replay_ids: Sequence[str],
    planned_looks: Sequence[int],
    prompt_bank_info: Mapping[str, Any] | None = None,
    russian_questions: Sequence[Question] = (),
) -> Path:
    script_path = Path(__file__).resolve()
    payload = {
        "code_version": CODE_VERSION,
        "script_filename": script_path.name,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "replay_version": REPLAY_VERSION,
        "adaptive_version": ADAPTIVE_VERSION,
        "pilot_data_included": False,
        "prompt_bank": dict(prompt_bank_info or {}),
        "primary_subset": PRIMARY_SUBSET,
        "russian_validation_subset": RUSSIAN_VALIDATION_SUBSET,
        "russian_validation_question_ids": [q.question_id for q in russian_questions],
        "seed": args.seed,
        "min_cases": args.min_cases,
        "batch_size": args.batch_size,
        "max_cases": args.max_cases,
        "planned_looks": list(planned_looks),
        "familywise_sequential_alpha": args.alpha,
        "per_look_boundary": args.alpha / len(planned_looks),
        "stopping_rule": (
            "At a planned look, stop for success only if all three prespecified "
            "component tests cross alpha/K, where K is the total number of planned "
            "looks. This Bonferroni-across-looks rule controls optional stopping "
            "conservatively. Otherwise continue to max_cases."
        ),
        "primary_endpoint": args.primary_endpoint,
        "primary_model": args.primary_model,
        "primary_rho0": args.rho0,
        "state_model": args.state_model,
        "minimum_state_shift": args.min_state_shift,
        "minimum_model_difference": args.min_model_difference,
        "external_ratings": str(args.external_ratings) if args.external_ratings else None,
        "replay_fraction": args.replay_fraction,
        "temperature": args.temperature,
        "workers": args.workers,
        "per_model_workers": args.per_model_workers,
        "retries": args.retries,
        "max_retry_delay": args.max_retry_delay,
        "live_max_tokens": args.live_max_tokens,
        "post_max_tokens": args.post_max_tokens,
        "replay_max_tokens": args.replay_max_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "ai_score_instruction": AI_SCORE_INSTRUCTION,
        "models": models,
        "provider_policy": "OpenRouter automatic provider routing with fallbacks enabled",
        "model_min_interval_seconds": MODEL_MIN_INTERVAL_SECONDS,
        "replay_question_ids_at_max": list(replay_ids),
        "questions_at_max": [
            {
                "question_id": q.question_id,
                "family": q.family,
                "task": q.task,
                "metadata": q.metadata,
            }
            for q in questions
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def adaptive_self_test() -> None:
    self_test()
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [1, 2, 3, 4, 5, 6, 7, 8]
    r, p = _spearman_greater_p(xs, ys, 0.2)
    assert r is not None and r > 0.99 and p is not None and p < 0.01
    n, mean_v, median_v, p_sign = _sign_test_greater([10.0] * 10, 5.0)
    assert n == 10 and mean_v == 10.0 and median_v == 10.0 and p_sign < 0.01
    qs = generate_questions(30, 20260906)
    sel = select_replay_questions_batched(qs, fraction=0.10, batch_size=10, seed=20260906)
    assert len(sel) == 3
    assert sel == select_replay_questions_batched(qs, fraction=0.10, batch_size=10, seed=20260906)
    assert _planned_looks(30, 120, 10) == list(range(30, 121, 10))
    print("adaptive self-test: all checks passed")



# ---------------------------------------------------------------------------
# V5.6 frozen-workspace extension:
#   1) full replay for GPT-5.5 + Gemini on all frozen confirmatory questions
#   2) paired score-guided vs random-target correction on the same frozen text
#   3) conservative minor repair of already-failed secondary cells
#
# IMPORTANT: this layer never regenerates a baseline cell whose LIVE status is
# already complete. Successful confirmatory baseline generations remain frozen.
# ---------------------------------------------------------------------------

DEFAULT_EXTENSION_MODELS = ("gpt-5.5", "gemini-3.1-pro-preview")


def _extension_schema(store: Store) -> None:
    with store.connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS extension_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extension_repairs (
                repair_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                repair_kind TEXT NOT NULL,
                previous_error TEXT,
                status TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, question_id, condition, repair_kind)
            );

            CREATE TABLE IF NOT EXISTS replay_extension_steps (
                model_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                sentence_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                score INTEGER,
                parse_mode TEXT,
                prompt_sha256 TEXT,
                source TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(model_id, question_id, sentence_index)
            );

            CREATE TABLE IF NOT EXISTS correction_trials (
                model_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                arm TEXT NOT NULL CHECK(arm IN ('guided','random')),
                target_sentence_index INTEGER NOT NULL,
                target_live_score INTEGER NOT NULL,
                source_sentence TEXT NOT NULL,
                status TEXT NOT NULL,
                revised_sentence TEXT,
                corrected_text TEXT,
                corrected_template TEXT,
                prompt_sha256 TEXT,
                provider TEXT,
                latency_s REAL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(model_id, question_id, arm)
            );

            CREATE TABLE IF NOT EXISTS correction_assessments (
                model_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                arm TEXT NOT NULL CHECK(arm IN ('guided','random')),
                status TEXT NOT NULL,
                scores_json TEXT,
                mean_score REAL,
                prompt_sha256 TEXT,
                provider TEXT,
                latency_s REAL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(model_id, question_id, arm)
            );
            """
        )


def _extension_write_meta(store: Store, key: str, value: Any) -> None:
    encoded = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True
    )
    with store.connect() as con:
        row = con.execute(
            "SELECT value FROM extension_meta WHERE key=?", (key,)
        ).fetchone()
        if row is not None and str(row["value"]) != encoded:
            raise ValueError(
                f"extension metadata {key!r} changed; use the same V5.6 "
                "extension settings when resuming"
            )
        con.execute(
            "INSERT OR IGNORE INTO extension_meta(key,value) VALUES (?,?)",
            (key, encoded),
        )


def _load_workspace_questions(store: Store) -> list[Question]:
    with store.connect() as con:
        rows = con.execute(
            "SELECT question_id,family,task,metadata_json FROM questions"
        ).fetchall()
    questions = [
        Question(
            question_id=str(r["question_id"]),
            family=str(r["family"]),
            task=str(r["task"]),
            metadata=json.loads(str(r["metadata_json"])),
        )
        for r in rows
    ]
    questions.sort(key=lambda q: int(q.metadata.get("case_index", 10**9)))
    return questions


def _workspace_routes(store: Store) -> dict[str, str]:
    with store.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT model_id,model_route FROM cells ORDER BY model_id"
        ).fetchall()
    routes: dict[str, str] = {}
    for r in rows:
        model_id = str(r["model_id"])
        route = str(r["model_route"])
        if model_id in routes and routes[model_id] != route:
            raise ValueError(f"workspace has multiple routes for {model_id}")
        routes[model_id] = route
    return routes


def _latest_raw_response(
    store: Store, model_id: str, qid: str, condition: str
) -> sqlite3.Row | None:
    with store.connect() as con:
        return con.execute(
            """
            SELECT response_text,response_json,error,provider,latency_s
            FROM api_calls
            WHERE model_id=? AND question_id=? AND condition=?
                  AND response_text IS NOT NULL
            ORDER BY call_id DESC
            LIMIT 1
            """,
            (model_id, qid, condition),
        ).fetchone()


def _record_repair(
    store: Store,
    *,
    model_id: str,
    qid: str,
    condition: str,
    repair_kind: str,
    previous_error: str | None,
    status: str,
    details: str,
) -> None:
    with store.connect() as con:
        con.execute(
            """
            INSERT INTO extension_repairs(
                model_id,question_id,condition,repair_kind,previous_error,
                status,details
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(model_id,question_id,condition,repair_kind)
            DO UPDATE SET
                previous_error=excluded.previous_error,
                status=excluded.status,
                details=excluded.details,
                created_at=CURRENT_TIMESTAMP
            """,
            (
                model_id, qid, condition, repair_kind, previous_error,
                status, details,
            ),
        )


def _repair_stored_parse_failures(store: Store) -> tuple[int, list[tuple[str, str]]]:
    """
    Re-parse already stored provider responses with V5.6 syntax tolerance.

    No API calls are made. This is appropriate only when the provider response
    was already generated and the old parser was unnecessarily strict.
    """
    repaired = 0
    repaired_live: list[tuple[str, str]] = []

    with store.connect() as con:
        rows = con.execute(
            """
            SELECT model_id,question_id,live_error,post_error,
                   live_status,post_status
            FROM cells
            WHERE live_status='error' OR post_status='error'
            ORDER BY model_id,question_id
            """
        ).fetchall()

    for row in rows:
        model_id = str(row["model_id"])
        qid = str(row["question_id"])

        if row["live_status"] == "error":
            raw = _latest_raw_response(store, model_id, qid, "live")
            if raw is not None and raw["response_text"]:
                try:
                    parsed = parse_live(
                        str(raw["response_text"]),
                        expected_min_scores=CASE_SENTENCE_MIN,
                        expected_max_scores=CASE_SENTENCE_MAX,
                    )
                except Exception:
                    pass
                else:
                    store.save_live(model_id, qid, parsed)
                    _record_repair(
                        store,
                        model_id=model_id,
                        qid=qid,
                        condition="live",
                        repair_kind="stored_reparse",
                        previous_error=row["live_error"],
                        status="complete",
                        details="Recovered from previously stored raw response; no new generation.",
                    )
                    repaired += 1
                    repaired_live.append((model_id, qid))

        # POST can be re-parsed only when LIVE is now complete.
        try:
            live_status = store.status(model_id, qid, "live")
        except KeyError:
            live_status = "missing"
        if row["post_status"] == "error" and live_status == "complete":
            raw = _latest_raw_response(store, model_id, qid, "post")
            if raw is not None and raw["response_text"]:
                try:
                    template = store.live_template(model_id, qid)
                    scores = parse_post(str(raw["response_text"]), template)
                except Exception:
                    pass
                else:
                    store.save_post(model_id, qid, str(raw["response_text"]), scores)
                    _record_repair(
                        store,
                        model_id=model_id,
                        qid=qid,
                        condition="post",
                        repair_kind="stored_reparse",
                        previous_error=row["post_error"],
                        status="complete",
                        details="Recovered from previously stored raw response; no new API call.",
                    )
                    repaired += 1

    return repaired, repaired_live


def _is_small_transient_failure(error: str | None) -> bool:
    if not error:
        return False
    e = error.lower()
    if "content_filter" in e:
        return False
    # Provider/network failures only. Parser/protocol failures are intentionally
    # not regenerated: they are model behavior.
    return (
        "httperror" in e
        or "urlerror" in e
        or "timeout" in e
        or "finish_reason='error'" in e
        or 'finish_reason="error"' in e
        or "finish_reason='stop'" in e
        or 'finish_reason="stop"' in e
    )


def _retry_minor_provider_failures(
    *,
    store: Store,
    client: FastFailOpenRouterClient,
    questions: Mapping[str, Question],
    routes: Mapping[str, str],
    repaired_live: Sequence[tuple[str, str]],
    live_max_tokens: int,
    post_max_tokens: int,
) -> int:
    """
    Retry only small provider/infrastructure failures and POST cells newly
    unlocked by a stored LIVE reparse. Never touch an already-complete LIVE.
    """
    repaired = 0
    newly_unlocked = set(repaired_live)

    with store.connect() as con:
        rows = con.execute(
            """
            SELECT model_id,question_id,live_status,live_error,
                   post_status,post_error
            FROM cells
            ORDER BY model_id,question_id
            """
        ).fetchall()

    for row in rows:
        model_id = str(row["model_id"])
        qid = str(row["question_id"])
        # In V5.7 a workspace may also contain the fixed Russian validation
        # subset. A repair pass scoped to English must ignore out-of-scope cells.
        if qid not in questions or model_id not in routes:
            continue
        q = questions[qid]
        route = routes[model_id]

        if (
            row["live_status"] == "error"
            and _is_small_transient_failure(row["live_error"])
        ):
            # One repair operation; the client itself has bounded transient retry.
            messages = live_messages(q.task)
            prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            result = None
            try:
                result = client.call(
                    model_id=model_id,
                    model_route=route,
                    messages=messages,
                    max_tokens=live_max_tokens,
                )
                parsed = parse_live(
                    result.text,
                    expected_min_scores=CASE_SENTENCE_MIN,
                    expected_max_scores=CASE_SENTENCE_MAX,
                )
                store.save_live(model_id, qid, parsed)
                store.record_call(
                    model_id=model_id, qid=qid, condition="repair_live",
                    sentence_index=None, prompt_hash=prompt_hash,
                    result=result, error=None,
                )
                _record_repair(
                    store, model_id=model_id, qid=qid, condition="live",
                    repair_kind="transient_retry",
                    previous_error=row["live_error"], status="complete",
                    details="Regenerated only because the original cell failed at provider/transport level.",
                )
                repaired += 1
                newly_unlocked.add((model_id, qid))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                store.record_call(
                    model_id=model_id, qid=qid, condition="repair_live",
                    sentence_index=None, prompt_hash=prompt_hash,
                    result=result, error=error,
                )
                _record_repair(
                    store, model_id=model_id, qid=qid, condition="live",
                    repair_kind="transient_retry",
                    previous_error=row["live_error"], status="error",
                    details=error,
                )

        # Retry POST only if it was a provider failure OR if a stored LIVE
        # reparse has just made a previously-pending POST eligible.
        live_now = store.status(model_id, qid, "live")
        post_now = store.status(model_id, qid, "post")
        should_post = (
            live_now == "complete"
            and post_now != "complete"
            and (
                _is_small_transient_failure(row["post_error"])
                or (model_id, qid) in newly_unlocked
            )
        )
        if should_post:
            template = store.live_template(model_id, qid)
            messages = post_messages(q.task, template)
            prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            result = None
            try:
                result = client.call(
                    model_id=model_id,
                    model_route=route,
                    messages=messages,
                    max_tokens=post_max_tokens,
                )
                scores = parse_post(result.text, template)
                store.save_post(model_id, qid, result.text, scores)
                store.record_call(
                    model_id=model_id, qid=qid, condition="repair_post",
                    sentence_index=None, prompt_hash=prompt_hash,
                    result=result, error=None,
                )
                _record_repair(
                    store, model_id=model_id, qid=qid, condition="post",
                    repair_kind="transient_or_unlocked_retry",
                    previous_error=row["post_error"], status="complete",
                    details="Post-hoc repair; baseline LIVE was not regenerated.",
                )
                repaired += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                store.record_call(
                    model_id=model_id, qid=qid, condition="repair_post",
                    sentence_index=None, prompt_hash=prompt_hash,
                    result=result, error=error,
                )
                _record_repair(
                    store, model_id=model_id, qid=qid, condition="post",
                    repair_kind="transient_or_unlocked_retry",
                    previous_error=row["post_error"], status="error",
                    details=error,
                )

    return repaired


def _baseline_replay_prefix_scores(
    store: Store, model_id: str, qid: str
) -> list[tuple[int, int, str]]:
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT sentence_index,score,COALESCE(parse_mode,'baseline') AS parse_mode
            FROM replay_steps
            WHERE model_id=? AND question_id=? AND status='complete'
            ORDER BY sentence_index
            """,
            (model_id, qid),
        ).fetchall()
    result: list[tuple[int, int, str]] = []
    for expected, row in enumerate(rows):
        idx = int(row["sentence_index"])
        if idx != expected:
            break
        result.append((idx, int(row["score"]), str(row["parse_mode"])))
    return result


def _seed_extension_replay_from_baseline(
    store: Store, model_id: str, qid: str
) -> None:
    baseline = _baseline_replay_prefix_scores(store, model_id, qid)
    if not baseline:
        return
    with store.connect() as con:
        for idx, score, parse_mode in baseline:
            con.execute(
                """
                INSERT OR IGNORE INTO replay_extension_steps(
                    model_id,question_id,sentence_index,status,score,parse_mode,
                    prompt_sha256,source,error
                ) VALUES (?, ?, ?, 'complete', ?, ?, NULL, 'baseline_replay', NULL)
                """,
                (model_id, qid, idx, score, parse_mode),
            )


def _extension_replay_scores(
    store: Store, model_id: str, qid: str
) -> list[int]:
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT sentence_index,score
            FROM replay_extension_steps
            WHERE model_id=? AND question_id=? AND status='complete'
            ORDER BY sentence_index
            """,
            (model_id, qid),
        ).fetchall()
    scores: list[int] = []
    for expected, row in enumerate(rows):
        idx = int(row["sentence_index"])
        if idx != expected:
            raise RuntimeError(
                f"non-contiguous extension replay for {(model_id, qid)}: "
                f"expected {expected}, got {idx}"
            )
        scores.append(int(row["score"]))
    return scores


def _save_extension_replay_success(
    store: Store,
    *,
    model_id: str,
    qid: str,
    sentence_index: int,
    score: int,
    parse_mode: str,
    prompt_hash: str,
) -> None:
    with store.connect() as con:
        con.execute(
            """
            INSERT INTO replay_extension_steps(
                model_id,question_id,sentence_index,status,score,parse_mode,
                prompt_sha256,source,error
            ) VALUES (?, ?, ?, 'complete', ?, ?, ?, 'v5.6_extension', NULL)
            ON CONFLICT(model_id,question_id,sentence_index) DO UPDATE SET
                status='complete',score=excluded.score,
                parse_mode=excluded.parse_mode,prompt_sha256=excluded.prompt_sha256,
                source=excluded.source,error=NULL,created_at=CURRENT_TIMESTAMP
            """,
            (model_id, qid, sentence_index, score, parse_mode, prompt_hash),
        )


def _save_extension_replay_error(
    store: Store,
    *,
    model_id: str,
    qid: str,
    sentence_index: int,
    prompt_hash: str,
    error: str,
) -> None:
    with store.connect() as con:
        con.execute(
            """
            INSERT INTO replay_extension_steps(
                model_id,question_id,sentence_index,status,score,parse_mode,
                prompt_sha256,source,error
            ) VALUES (?, ?, ?, 'error', NULL, NULL, ?, 'v5.6_extension', ?)
            ON CONFLICT(model_id,question_id,sentence_index) DO UPDATE SET
                status='error',score=NULL,parse_mode=NULL,
                prompt_sha256=excluded.prompt_sha256,source=excluded.source,
                error=excluded.error,created_at=CURRENT_TIMESTAMP
            """,
            (model_id, qid, sentence_index, prompt_hash, error),
        )


def _live_score_vector(store: Store, model_id: str, qid: str) -> list[int]:
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT sentence_index,score FROM scores
            WHERE model_id=? AND question_id=? AND condition='live'
            ORDER BY sentence_index
            """,
            (model_id, qid),
        ).fetchall()
    scores = [int(r["score"]) for r in rows]
    for expected, row in enumerate(rows):
        if int(row["sentence_index"]) != expected:
            raise RuntimeError(f"non-contiguous LIVE scores for {(model_id, qid)}")
    return scores


def _split_piece_whitespace(piece: str) -> tuple[str, str, str]:
    m1 = re.match(r"^\s*", piece)
    m2 = re.search(r"\s*$", piece)
    lead = m1.group(0) if m1 else ""
    trail = m2.group(0) if m2 else ""
    end = len(piece) - len(trail) if trail else len(piece)
    core = piece[len(lead):end]
    return lead, core, trail


def _replace_template_sentence(
    template: str, sentence_index: int, replacement: str
) -> tuple[str, str]:
    pieces = split_template(template)
    n = len(pieces) - 1
    if not 0 <= sentence_index < n:
        raise IndexError(sentence_index)
    sentence_pieces = list(pieces[:-1])
    lead, _core, trail = _split_piece_whitespace(sentence_pieces[sentence_index])
    sentence_pieces[sentence_index] = lead + replacement.strip() + trail
    corrected_template = "".join(
        piece + SLOT for piece in sentence_pieces
    ) + pieces[-1]
    corrected_text = corrected_template.replace(SLOT, "")
    return corrected_text, corrected_template


def _plain_frozen_text(template: str) -> str:
    return template.replace(SLOT, "")


def correction_messages(
    task: str,
    full_frozen_text: str,
    target_sentence: str,
    *,
    language: str = "en",
) -> list[dict[str, str]]:
    if language == "ru":
        instruction = (
            "Перепиши только ЦЕЛЕВОЕ ПРЕДЛОЖЕНИЕ так, чтобы обычный читатель "
            "с меньшей вероятностью решил, что текст написан ИИ. Сохрани смысл "
            "и фактическое содержание, оставь это одним предложением и сделай "
            "так, чтобы оно естественно вписывалось в остальной текст. Не "
            "переписывай другие предложения. Верни только новое предложение: "
            "без оценки, подписи, объяснения, кавычек вокруг всего ответа или "
            "дополнительного текста."
        )
        headings = (
            "ИСХОДНОЕ ЗАДАНИЕ",
            "ЗАМОРОЖЕННЫЙ СГЕНЕРИРОВАННЫЙ ТЕКСТ",
            "ЦЕЛЕВОЕ ПРЕДЛОЖЕНИЕ",
        )
    else:
        instruction = (
            "Rewrite only the TARGET SENTENCE so that an average human reader "
            "would be less likely to think it was AI-generated. Preserve its "
            "intended meaning and factual content, keep it as one sentence, and "
            "make it fit naturally into the frozen text. Do not rewrite any "
            "other sentence. Return only the replacement sentence, with no "
            "score, label, explanation, quotation marks around the whole answer, "
            "or additional text."
        )
        headings = (
            "ORIGINAL WRITING TASK",
            "FROZEN GENERATED TEXT",
            "TARGET SENTENCE",
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{headings[0]}\n{'-' * len(headings[0])}\n{task}\n\n"
                f"{headings[1]}\n{'-' * len(headings[1])}\n{full_frozen_text}\n\n"
                f"{headings[2]}\n{'-' * len(headings[2])}\n{target_sentence}\n\n"
                f"{instruction}"
            ),
        },
    ]


def _parse_rewritten_sentence(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        raise ValueError("empty correction response")
    if "\n" in candidate or "\r" in candidate:
        raise ValueError("correction response must be exactly one line/sentence")
    if _SCORE_LIKE_RE.search(candidate) or _BARE_SCORE_RE.search(candidate):
        raise ValueError("correction response unexpectedly contains an AI score")
    if len(candidate) > 1200:
        raise ValueError("correction response is implausibly long")
    lower = candidate.lower()
    if lower.startswith(("rewritten sentence:", "replacement:", "here is", "here's")):
        raise ValueError("correction response contains explanatory wrapper text")
    return candidate


def _deterministic_random_target(
    *,
    n_sentences: int,
    guided_index: int,
    seed: int,
    model_id: str,
    qid: str,
) -> int:
    choices = [i for i in range(n_sentences) if i != guided_index]
    if not choices:
        raise ValueError("need at least two sentences for paired correction")
    material = f"{seed}|{model_id}|{qid}|random-control"
    derived = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(derived)
    return rng.choice(choices)


class ExtensionRunner(Runner):
    def run_full_replay(self, qids: Sequence[str]) -> None:
        jobs = []
        for qid in qids:
            q = self.qmap[qid]
            for model_id, route in self.models.items():
                if self.store.status(model_id, qid, "live") != "complete":
                    continue
                if self.store.status(model_id, qid, "post") != "complete":
                    continue
                _seed_extension_replay_from_baseline(self.store, model_id, qid)

                def job(model_id=model_id, route=route, q=q):
                    template = self.store.live_template(model_id, q.question_id)
                    n_sentences = len(split_template(template)) - 1
                    replay_scores = _extension_replay_scores(
                        self.store, model_id, q.question_id
                    )
                    if len(replay_scores) >= n_sentences:
                        return
                    for sentence_index in range(len(replay_scores), n_sentences):
                        prefix = build_replay_prefix(
                            template, replay_scores, sentence_index
                        )
                        messages = replay_messages(q.task, prefix)
                        prompt_hash = self._prompt_hash(messages)
                        result = None
                        try:
                            result = self._call_model(
                                model_id=model_id,
                                model_route=route,
                                messages=messages,
                                max_tokens=self.replay_max_tokens,
                            )
                            score, parse_mode = parse_single_replay_score(
                                result.text, replay_scores
                            )
                            _save_extension_replay_success(
                                self.store,
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                score=score,
                                parse_mode=parse_mode,
                                prompt_hash=prompt_hash,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay_ext",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=None,
                            )
                            replay_scores.append(score)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            _save_extension_replay_error(
                                self.store,
                                model_id=model_id,
                                qid=q.question_id,
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                error=error,
                            )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition="replay_ext",
                                sentence_index=sentence_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=error,
                            )
                            raise

                jobs.append((model_id, qid, job))
        self._run_parallel(jobs, "replay-ext")

    def run_corrections(self, qids: Sequence[str], correction_seed: int) -> None:
        jobs = []
        for qid in qids:
            q = self.qmap[qid]
            for model_id, route in self.models.items():
                if self.store.status(model_id, qid, "live") != "complete":
                    continue
                template = self.store.live_template(model_id, qid)
                pieces = split_template(template)
                n_sentences = len(pieces) - 1
                live_scores = _live_score_vector(self.store, model_id, qid)
                if len(live_scores) != n_sentences:
                    continue
                guided_index = max(
                    range(n_sentences),
                    key=lambda i: (live_scores[i], -i),
                )
                random_index = _deterministic_random_target(
                    n_sentences=n_sentences,
                    guided_index=guided_index,
                    seed=correction_seed,
                    model_id=model_id,
                    qid=qid,
                )
                targets = {"guided": guided_index, "random": random_index}

                for arm, target_index in targets.items():
                    with self.store.connect() as con:
                        existing = con.execute(
                            """
                            SELECT status FROM correction_trials
                            WHERE model_id=? AND question_id=? AND arm=?
                            """,
                            (model_id, qid, arm),
                        ).fetchone()
                    if existing is not None and existing["status"] == "complete":
                        continue

                    def job(
                        model_id=model_id,
                        route=route,
                        q=q,
                        template=template,
                        pieces=pieces,
                        live_scores=live_scores,
                        arm=arm,
                        target_index=target_index,
                    ):
                        _lead, source_sentence, _trail = _split_piece_whitespace(
                            pieces[target_index]
                        )
                        full_text = _plain_frozen_text(template)
                        messages = correction_messages(
                            q.task, full_text, source_sentence.strip(),
                            language=str(q.metadata.get("language", "en")),
                        )
                        prompt_hash = self._prompt_hash(messages)
                        result = None
                        try:
                            result = self._call_model(
                                model_id=model_id,
                                model_route=route,
                                messages=messages,
                                max_tokens=min(self.live_max_tokens, 1536),
                            )
                            revised = _parse_rewritten_sentence(result.text)
                            corrected_text, corrected_template = _replace_template_sentence(
                                template, target_index, revised
                            )
                            with self.store.connect() as con:
                                con.execute(
                                    """
                                    INSERT INTO correction_trials(
                                        model_id,question_id,arm,
                                        target_sentence_index,target_live_score,
                                        source_sentence,status,revised_sentence,
                                        corrected_text,corrected_template,
                                        prompt_sha256,provider,latency_s,error
                                    ) VALUES (?,?,?,?,?,?,'complete',?,?,?,?,?,?,NULL)
                                    ON CONFLICT(model_id,question_id,arm) DO UPDATE SET
                                        target_sentence_index=excluded.target_sentence_index,
                                        target_live_score=excluded.target_live_score,
                                        source_sentence=excluded.source_sentence,
                                        status='complete',
                                        revised_sentence=excluded.revised_sentence,
                                        corrected_text=excluded.corrected_text,
                                        corrected_template=excluded.corrected_template,
                                        prompt_sha256=excluded.prompt_sha256,
                                        provider=excluded.provider,
                                        latency_s=excluded.latency_s,
                                        error=NULL,
                                        created_at=CURRENT_TIMESTAMP
                                    """,
                                    (
                                        model_id, q.question_id, arm,
                                        target_index, int(live_scores[target_index]),
                                        source_sentence.strip(), revised,
                                        corrected_text, corrected_template,
                                        prompt_hash, result.provider, result.latency_s,
                                    ),
                                )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition=f"correction_{arm}",
                                sentence_index=target_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=None,
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            with self.store.connect() as con:
                                con.execute(
                                    """
                                    INSERT INTO correction_trials(
                                        model_id,question_id,arm,
                                        target_sentence_index,target_live_score,
                                        source_sentence,status,error
                                    ) VALUES (?,?,?,?,?,?,'error',?)
                                    ON CONFLICT(model_id,question_id,arm) DO UPDATE SET
                                        status='error',error=excluded.error,
                                        created_at=CURRENT_TIMESTAMP
                                    """,
                                    (
                                        model_id, q.question_id, arm,
                                        target_index, int(live_scores[target_index]),
                                        source_sentence.strip(), error,
                                    ),
                                )
                            self.store.record_call(
                                model_id=model_id,
                                qid=q.question_id,
                                condition=f"correction_{arm}",
                                sentence_index=target_index,
                                prompt_hash=prompt_hash,
                                result=result,
                                error=error,
                            )
                            raise

                    jobs.append((model_id, f"{qid}:{arm}", job))

        self._run_parallel(jobs, "correction")

    def run_correction_assessments(self) -> None:
        jobs = []
        with self.store.connect() as con:
            rows = con.execute(
                """
                SELECT t.model_id,t.question_id,t.arm,t.corrected_template
                FROM correction_trials t
                LEFT JOIN correction_assessments a
                  ON a.model_id=t.model_id
                 AND a.question_id=t.question_id
                 AND a.arm=t.arm
                WHERE t.status='complete'
                  AND (a.status IS NULL OR a.status!='complete')
                ORDER BY t.question_id,t.model_id,t.arm
                """
            ).fetchall()

        for row in rows:
            model_id = str(row["model_id"])
            if model_id not in self.models:
                continue
            qid = str(row["question_id"])
            arm = str(row["arm"])
            template = str(row["corrected_template"])
            q = self.qmap[qid]
            route = self.models[model_id]

            def job(
                model_id=model_id, qid=qid, arm=arm,
                template=template, q=q, route=route,
            ):
                messages = post_messages(q.task, template)
                prompt_hash = self._prompt_hash(messages)
                result = None
                try:
                    result = self._call_model(
                        model_id=model_id,
                        model_route=route,
                        messages=messages,
                        max_tokens=self.post_max_tokens,
                    )
                    scores = parse_post(result.text, template)
                    mean_score = statistics.fmean(scores)
                    with self.store.connect() as con:
                        con.execute(
                            """
                            INSERT INTO correction_assessments(
                                model_id,question_id,arm,status,scores_json,
                                mean_score,prompt_sha256,provider,latency_s,error
                            ) VALUES (?, ?, ?, 'complete', ?, ?, ?, ?, ?, NULL)
                            ON CONFLICT(model_id,question_id,arm) DO UPDATE SET
                                status='complete',scores_json=excluded.scores_json,
                                mean_score=excluded.mean_score,
                                prompt_sha256=excluded.prompt_sha256,
                                provider=excluded.provider,
                                latency_s=excluded.latency_s,
                                error=NULL,created_at=CURRENT_TIMESTAMP
                            """,
                            (
                                model_id, qid, arm, json.dumps(list(scores)),
                                mean_score, prompt_hash,
                                result.provider, result.latency_s,
                            ),
                        )
                    self.store.record_call(
                        model_id=model_id,
                        qid=qid,
                        condition=f"correction_post_{arm}",
                        sentence_index=None,
                        prompt_hash=prompt_hash,
                        result=result,
                        error=None,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    with self.store.connect() as con:
                        con.execute(
                            """
                            INSERT INTO correction_assessments(
                                model_id,question_id,arm,status,error
                            ) VALUES (?, ?, ?, 'error', ?)
                            ON CONFLICT(model_id,question_id,arm) DO UPDATE SET
                                status='error',error=excluded.error,
                                created_at=CURRENT_TIMESTAMP
                            """,
                            (model_id, qid, arm, error),
                        )
                    self.store.record_call(
                        model_id=model_id,
                        qid=qid,
                        condition=f"correction_post_{arm}",
                        sentence_index=None,
                        prompt_hash=prompt_hash,
                        result=result,
                        error=error,
                    )
                    raise

            jobs.append((model_id, f"{qid}:{arm}", job))

        self._run_parallel(jobs, "correction-post")


def _export_replay_extension(
    out_dir: Path, store: Store, models: Sequence[str]
) -> dict[str, Any]:
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT model_id,question_id,sentence_index,score,source,status,error
            FROM replay_extension_steps
            WHERE model_id IN ({})
            ORDER BY model_id,question_id,sentence_index
            """.format(",".join("?" for _ in models)),
            tuple(models),
        ).fetchall()

    _write_csv(
        out_dir / "expanded_replay_scores.csv",
        [
            "model_id","question_id","sentence_index","score",
            "source","status","error"
        ],
        [
            [
                r["model_id"],r["question_id"],r["sentence_index"],r["score"],
                r["source"],r["status"],r["error"]
            ]
            for r in rows
        ],
    )

    summaries: list[dict[str, Any]] = []
    for model_id in models:
        live = _question_means(store, model_id, "live")
        post = _question_means(store, model_id, "post")
        with store.connect() as con:
            rr = con.execute(
                """
                SELECT question_id,AVG(score) AS mean_score
                FROM replay_extension_steps
                WHERE model_id=? AND status='complete'
                GROUP BY question_id
                """,
                (model_id,),
            ).fetchall()
            counts = {
                str(r["question_id"]): int(r["n"])
                for r in con.execute(
                    """
                    SELECT question_id,COUNT(*) AS n
                    FROM replay_extension_steps
                    WHERE model_id=? AND status='complete'
                    GROUP BY question_id
                    """,
                    (model_id,),
                ).fetchall()
            }
        replay = {str(r["question_id"]): float(r["mean_score"]) for r in rr}

        # Require a complete replay: score count equals number of LIVE score slots.
        complete_qids = []
        for qid in sorted(live.keys() & post.keys() & replay.keys()):
            try:
                n_expected = len(split_template(store.live_template(model_id, qid))) - 1
            except KeyError:
                continue
            if counts.get(qid) == n_expected:
                complete_qids.append(qid)

        lv = [live[q] for q in complete_qids]
        po = [post[q] for q in complete_qids]
        rp = [replay[q] for q in complete_qids]
        summaries.append(
            {
                "model_id": model_id,
                "n_questions": len(complete_qids),
                "mean_live": statistics.fmean(lv) if lv else None,
                "mean_post": statistics.fmean(po) if po else None,
                "mean_replay": statistics.fmean(rp) if rp else None,
                "live_replay_spearman": spearman(lv, rp) if len(lv) >= 2 else None,
                "post_replay_spearman": spearman(po, rp) if len(po) >= 2 else None,
                "live_replay_mae": (
                    statistics.fmean(abs(a-b) for a,b in zip(lv,rp))
                    if lv else None
                ),
                "post_replay_mae": (
                    statistics.fmean(abs(a-b) for a,b in zip(po,rp))
                    if po else None
                ),
            }
        )

    fields = [
        "model_id","n_questions","mean_live","mean_post","mean_replay",
        "live_replay_spearman","post_replay_spearman",
        "live_replay_mae","post_replay_mae",
    ]
    _write_csv(
        out_dir / "expanded_replay_summary.csv",
        fields,
        [[s.get(f) for f in fields] for s in summaries],
    )
    return {"models": summaries}


def _export_correction_extension(
    out_dir: Path,
    store: Store,
    models: Sequence[str],
    blind_seed: int,
) -> dict[str, Any]:
    with store.connect() as con:
        trials = con.execute(
            """
            SELECT * FROM correction_trials
            WHERE model_id IN ({})
            ORDER BY model_id,question_id,arm
            """.format(",".join("?" for _ in models)),
            tuple(models),
        ).fetchall()
        assessments = con.execute(
            """
            SELECT * FROM correction_assessments
            WHERE model_id IN ({})
            ORDER BY model_id,question_id,arm
            """.format(",".join("?" for _ in models)),
            tuple(models),
        ).fetchall()

    trial_fields = [
        "model_id","question_id","arm","target_sentence_index",
        "target_live_score","source_sentence","status","revised_sentence",
        "corrected_text","provider","latency_s","error"
    ]
    _write_csv(
        out_dir / "correction_trials.csv",
        trial_fields,
        [[r[f] for f in trial_fields] for r in trials],
    )

    amap = {
        (str(r["model_id"]),str(r["question_id"]),str(r["arm"])): r
        for r in assessments if r["status"] == "complete"
    }
    tmap = {
        (str(r["model_id"]),str(r["question_id"]),str(r["arm"])): r
        for r in trials if r["status"] == "complete"
    }

    pair_rows = []
    summary_models = []
    for model_id in models:
        qids = sorted({
            q for (m,q,a) in tmap
            if m == model_id
            and (m,q,"guided") in tmap
            and (m,q,"random") in tmap
            and (m,q,"guided") in amap
            and (m,q,"random") in amap
        })
        advantages = []
        guided_changes = []
        random_changes = []
        for qid in qids:
            baseline_post = _question_means(store, model_id, "post").get(qid)
            if baseline_post is None:
                continue
            g = float(amap[(model_id,qid,"guided")]["mean_score"])
            r = float(amap[(model_id,qid,"random")]["mean_score"])
            g_change = g - baseline_post
            r_change = r - baseline_post
            advantage = r - g  # positive => guided final score is lower/better
            advantages.append(advantage)
            guided_changes.append(g_change)
            random_changes.append(r_change)
            pair_rows.append([
                model_id,qid,baseline_post,g,r,
                g_change,r_change,advantage,
                tmap[(model_id,qid,"guided")]["target_sentence_index"],
                tmap[(model_id,qid,"guided")]["target_live_score"],
                tmap[(model_id,qid,"random")]["target_sentence_index"],
                tmap[(model_id,qid,"random")]["target_live_score"],
            ])
        n_non_ties, mean_adv, median_adv, p_adv = _sign_test_greater(
            advantages, 0.0
        )
        summary_models.append(
            {
                "model_id": model_id,
                "n_paired_questions": len(advantages),
                "mean_guided_change_vs_baseline_post": (
                    statistics.fmean(guided_changes) if guided_changes else None
                ),
                "mean_random_change_vs_baseline_post": (
                    statistics.fmean(random_changes) if random_changes else None
                ),
                "mean_guided_advantage_random_minus_guided": (
                    statistics.fmean(advantages) if advantages else None
                ),
                "median_guided_advantage_random_minus_guided": (
                    statistics.median(advantages) if advantages else None
                ),
                "sign_test_n_non_ties": n_non_ties,
                "sign_test_p_guided_better": p_adv if advantages else None,
            }
        )

    pair_fields = [
        "model_id","question_id","baseline_post_mean",
        "guided_corrected_post_mean","random_corrected_post_mean",
        "guided_change_vs_baseline","random_change_vs_baseline",
        "guided_advantage_random_minus_guided",
        "guided_target_index","guided_target_live_score",
        "random_target_index","random_target_live_score",
    ]
    _write_csv(out_dir / "correction_pairwise.csv", pair_fields, pair_rows)

    summary_fields = [
        "model_id","n_paired_questions",
        "mean_guided_change_vs_baseline_post",
        "mean_random_change_vs_baseline_post",
        "mean_guided_advantage_random_minus_guided",
        "median_guided_advantage_random_minus_guided",
        "sign_test_n_non_ties","sign_test_p_guided_better",
    ]
    _write_csv(
        out_dir / "correction_summary.csv",
        summary_fields,
        [[s.get(f) for f in summary_fields] for s in summary_models],
    )

    # Human-evaluation artifacts. The public/blind file contains no model or arm.
    # The key is kept separately.
    blind_items = []
    blind_key = []
    pair_items = []
    pair_key = []

    with store.connect() as con:
        task_rows = {
            str(r["question_id"]): str(r["task"])
            for r in con.execute("SELECT question_id,task FROM questions")
        }

    for model_id in models:
        qids = sorted({
            q for (m,q,a) in tmap
            if m == model_id
            and (m,q,"guided") in tmap
            and (m,q,"random") in tmap
        })
        for qid in qids:
            baseline_template = store.live_template(model_id, qid)
            baseline_text = _plain_frozen_text(baseline_template)
            variants = {
                "baseline": baseline_text,
                "guided": str(tmap[(model_id,qid,"guided")]["corrected_text"]),
                "random": str(tmap[(model_id,qid,"random")]["corrected_text"]),
            }
            # Numeric blind items, randomized independently.
            order = list(variants)
            material = f"{blind_seed}|{model_id}|{qid}|items"
            rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16],16))
            rng.shuffle(order)
            for j, arm in enumerate(order):
                item_id = hashlib.sha256(
                    f"{blind_seed}|{model_id}|{qid}|{arm}".encode()
                ).hexdigest()[:16]
                blind_items.append([item_id, variants[arm], ""])
                blind_key.append([item_id, model_id, qid, arm])

            # Direct guided-vs-random paired preference.
            material = f"{blind_seed}|{model_id}|{qid}|pair"
            rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16],16))
            if rng.random() < 0.5:
                arm_a, arm_b = "guided", "random"
            else:
                arm_a, arm_b = "random", "guided"
            pair_id = hashlib.sha256(
                f"{blind_seed}|{model_id}|{qid}|pair".encode()
            ).hexdigest()[:16]
            pair_items.append([
                pair_id,
                variants[arm_a],
                variants[arm_b],
                "",
            ])
            pair_key.append([pair_id, model_id, qid, arm_a, arm_b])

    _write_csv(
        out_dir / "correction_blind_items.csv",
        ["item_id","text","human_ai_score_0_100"],
        blind_items,
    )
    _write_csv(
        out_dir / "correction_blind_key.csv",
        ["item_id","model_id","question_id","arm"],
        blind_key,
    )
    _write_csv(
        out_dir / "correction_blind_pairs.csv",
        ["pair_id","text_A","text_B","more_human_A_B_or_tie"],
        pair_items,
    )
    _write_csv(
        out_dir / "correction_blind_pair_key.csv",
        ["pair_id","model_id","question_id","arm_A","arm_B"],
        pair_key,
    )

    return {"models": summary_models}


def _write_extension_manifest(
    out_dir: Path,
    *,
    script_path: Path,
    script_sha: str,
    models: Sequence[str],
    replay_mode: str,
    correction_mode: str,
    correction_seed: int,
    repair_minor: bool,
) -> Path:
    path = out_dir / "extension_manifest_v56.json"
    payload = {
        "code_version": CODE_VERSION,
        "extension_version": EXTENSION_VERSION,
        "script_filename": script_path.name,
        "script_sha256": script_sha,
        "workspace": str(out_dir),
        "models": list(models),
        "expanded_replay": replay_mode,
        "self_correction": correction_mode,
        "correction_seed": correction_seed,
        "repair_minor": repair_minor,
        "baseline_policy": (
            "Never regenerate an already-complete LIVE baseline cell. "
            "Expanded replay and correction operate on frozen successful text."
        ),
        "correction_design": {
            "guided_arm": "rewrite the highest-LIVE-score sentence",
            "control_arm": "rewrite a deterministic random different sentence",
            "prompt_difference": "target sentence only; score value is hidden in both arms",
            "automatic_diagnostic": "same-model post assessment of both corrected outputs",
            "confirmatory_evaluation": "use blinded human artifacts",
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def extension_self_test() -> None:
    # Harmless whitespace inside opening angle bracket is accepted.
    x = (
        "One sentence. < AI SCORE: 12>\n"
        "Two sentence. <AI SCORE: 20>\n"
        "Three sentence. <AI SCORE: 30>\n"
        "Four sentence. <AI SCORE: 40>\n"
        "Five sentence. <AI SCORE: 50>\n"
        "Six sentence. <AI SCORE: 60>"
    )
    parsed = parse_live(x, expected_min_scores=6, expected_max_scores=8)
    assert parsed.scores[0] == 12

    t = (
        "One. <AI SCORE: ?>\nTwo. <AI SCORE: ?>\n"
        "Three. <AI SCORE: ?>\nFour. <AI SCORE: ?>\n"
        "Five. <AI SCORE: ?>\nSix. <AI SCORE: ?>"
    )
    corrected_text, corrected_template = _replace_template_sentence(
        t, 2, "Three, but rewritten."
    )
    assert corrected_template.count(SLOT) == 6
    assert "Three, but rewritten." in corrected_text

    a = _deterministic_random_target(
        n_sentences=7, guided_index=3, seed=20260906,
        model_id="gpt-5.5", qid="q0001"
    )
    b = _deterministic_random_target(
        n_sentences=7, guided_index=3, seed=20260906,
        model_id="gpt-5.5", qid="q0001"
    )
    assert a == b and a != 3
    print("V5.6 extension self-test: all checks passed")


def extension_main(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    script_path: Path,
    script_sha: str,
) -> int:
    db_path = args.out / "experiment.sqlite3"
    if not db_path.exists():
        parser.error(
            f"--extend-existing requires an existing workspace DB: {db_path}"
        )

    store = Store(db_path)
    _extension_schema(store)
    questions = _load_workspace_questions(store)
    qmap = {q.question_id: q for q in questions}
    routes = _workspace_routes(store)

    requested_models = [
        s.strip() for s in args.extension_models.split(",") if s.strip()
    ]
    if not requested_models:
        parser.error("--extension-models cannot be empty")
    unknown = [m for m in requested_models if m not in routes]
    if unknown:
        parser.error(f"models not present in workspace: {', '.join(unknown)}")
    models = {m: routes[m] for m in requested_models}

    load_simple_dotenv(Path.cwd() / ".env")
    load_simple_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        parser.error("OPENROUTER_API_KEY is not set")

    _extension_write_meta(store, "extension_version", EXTENSION_VERSION)
    _extension_write_meta(store, "extension_models", requested_models)
    _extension_write_meta(store, "correction_seed", args.correction_seed)
    _extension_write_meta(store, "expanded_replay", args.expanded_replay)
    _extension_write_meta(store, "self_correction", args.self_correction)

    manifest = _write_extension_manifest(
        args.out,
        script_path=script_path,
        script_sha=script_sha,
        models=requested_models,
        replay_mode=args.expanded_replay,
        correction_mode=args.self_correction,
        correction_seed=args.correction_seed,
        repair_minor=args.repair_minor,
    )

    print(f"V5.6 extension workspace: {args.out}", flush=True)
    print(f"Extension manifest: {manifest}", flush=True)
    print(f"Frozen baseline questions: {len(questions)}", flush=True)
    print(f"Extension models: {', '.join(requested_models)}", flush=True)
    print(
        "Baseline policy: completed LIVE generations are NEVER regenerated.",
        flush=True,
    )
    print(
        f"Expanded replay: {args.expanded_replay}; "
        f"self-correction: {args.self_correction}; "
        f"minor repair: {args.repair_minor}",
        flush=True,
    )

    if args.dry_run:
        with store.connect() as con:
            for model_id in requested_models:
                r = con.execute(
                    """
                    SELECT COUNT(*) AS n,
                           SUM(live_status='complete') AS live_n,
                           SUM(post_status='complete') AS post_n
                    FROM cells WHERE model_id=?
                    """,
                    (model_id,),
                ).fetchone()
                print(
                    f"  {model_id}: cells={r['n']} live={r['live_n']} post={r['post_n']}",
                    flush=True,
                )
        print("DRY RUN: no API calls made.", flush=True)
        return 0

    client = FastFailOpenRouterClient(
        api_key=api_key,
        temperature=args.temperature,
        timeout_s=args.timeout,
        retries=args.retries,
        max_retry_delay_s=args.max_retry_delay,
    )

    if args.repair_minor:
        reparsed, repaired_live = _repair_stored_parse_failures(store)
        retried = _retry_minor_provider_failures(
            store=store,
            client=client,
            questions=qmap,
            routes=routes,
            repaired_live=repaired_live,
            live_max_tokens=args.live_max_tokens,
            post_max_tokens=args.post_max_tokens,
        )
        print(
            f"[minor repair] stored-response reparses={reparsed}; "
            f"provider/unlocked retries completed={retried}",
            flush=True,
        )

    # The extension sample is the intersection of frozen successful LIVE+POST
    # cells for the two prespecified models. In the V5.5 confirmatory workspace
    # this is all 40 questions for GPT-5.5 and Gemini.
    eligible_sets = []
    for model_id in requested_models:
        with store.connect() as con:
            rows = con.execute(
                """
                SELECT question_id FROM cells
                WHERE model_id=? AND live_status='complete' AND post_status='complete'
                """,
                (model_id,),
            ).fetchall()
        eligible_sets.append({str(r["question_id"]) for r in rows})
    common_qids = sorted(
        set.intersection(*eligible_sets) if eligible_sets else set(),
        key=lambda qid: int(qmap[qid].metadata.get("case_index", 10**9)),
    )
    print(
        f"Matched frozen LIVE+POST questions available to extension: {len(common_qids)}",
        flush=True,
    )

    runner = ExtensionRunner(
        store=store,
        client=client,
        questions=[qmap[qid] for qid in common_qids],
        models=models,
        workers=args.extension_workers,
        per_model_workers=args.extension_per_model_workers,
        live_max_tokens=args.live_max_tokens,
        post_max_tokens=args.post_max_tokens,
        replay_max_tokens=args.replay_max_tokens,
    )

    if args.expanded_replay == "all":
        runner.run_full_replay(common_qids)

    if args.self_correction == "paired":
        runner.run_corrections(common_qids, args.correction_seed)
        if not args.no_correction_post_assess:
            runner.run_correction_assessments()

    replay_summary = _export_replay_extension(
        args.out, store, requested_models
    )
    correction_summary = _export_correction_extension(
        args.out, store, requested_models, args.correction_seed + 17
    )

    extension_summary = {
        "code_version": CODE_VERSION,
        "extension_version": EXTENSION_VERSION,
        "matched_questions": len(common_qids),
        "models": requested_models,
        "expanded_replay": replay_summary,
        "self_correction": correction_summary,
    }
    (args.out / "extension_summary_v56.json").write_text(
        json.dumps(extension_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    export_results(args.out, store)
    print("\nV5.6 extension finished.", flush=True)
    print(f"Expanded replay: {args.out / 'expanded_replay_summary.csv'}", flush=True)
    print(f"Correction summary: {args.out / 'correction_summary.csv'}", flush=True)
    print(f"Correction pairs: {args.out / 'correction_pairwise.csv'}", flush=True)
    print(f"Blind human items: {args.out / 'correction_blind_items.csv'}", flush=True)
    print(f"Blind A/B pairs: {args.out / 'correction_blind_pairs.csv'}", flush=True)
    print(
        "Keep correction_blind_key.csv and correction_blind_pair_key.csv away "
        "from raters.",
        flush=True,
    )
    return 0



def _paired_russian_sentence_candidates(
    store: Store,
    *,
    qids: set[str],
    model_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with store.connect() as con:
        cell_rows = con.execute(
            """
            SELECT question_id,live_template
            FROM cells
            WHERE model_id=? AND live_status='complete'
            ORDER BY question_id
            """,
            (model_id,),
        ).fetchall()
    for cell in cell_rows:
        qid = str(cell["question_id"])
        if qid not in qids:
            continue
        template = str(cell["live_template"])
        pieces = split_template(template)[:-1]
        scores = _live_score_vector(store, model_id, qid)
        if len(pieces) != len(scores):
            continue
        for idx, (piece, score) in enumerate(zip(pieces, scores)):
            sentence = piece.strip()
            if sentence:
                rows.append(
                    {
                        "model_id": model_id,
                        "question_id": qid,
                        "sentence_index": idx,
                        "sentence": sentence,
                        "live_score": float(score),
                    }
                )
    return rows


def export_russian_human_validation(
    out_dir: Path,
    store: Store,
    *,
    russian_qids: Sequence[str],
    models: Sequence[str],
    seed: int,
    preferred_min_gap: float,
) -> dict[str, Any]:
    """
    Export two blinded author-rating tasks:
      1) sentence pairs selected only from pre-existing LIVE-score separation;
      2) guided-vs-random corrected full posts.

    Keys are separate and must not be shown before ratings are frozen.
    """
    qset = set(russian_qids)
    blind_rows: list[list[Any]] = []
    key_rows: list[list[Any]] = []
    pair_counts: dict[str, int] = {}

    for model_id in models:
        candidates = _paired_russian_sentence_candidates(
            store, qids=qset, model_id=model_id
        )
        unused = list(candidates)
        chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
        # Greedy maximum-separation pairing, no sentence reuse and preferably
        # different questions. This selection uses model LIVE scores only, never
        # human or correction outcomes.
        while len(chosen) < 5 and len(unused) >= 2:
            best = None
            best_key = None
            for i, a in enumerate(unused):
                for j in range(i + 1, len(unused)):
                    b = unused[j]
                    if a["question_id"] == b["question_id"]:
                        continue
                    gap = abs(a["live_score"] - b["live_score"])
                    # Prefer gaps above threshold; within tier maximize gap.
                    tier = 1 if gap >= preferred_min_gap else 0
                    key = (tier, gap)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = (i, j, a, b)
            if best is None:
                break
            i, j, a, b = best
            chosen.append((a, b))
            for idx in sorted((i, j), reverse=True):
                unused.pop(idx)

        pair_counts[model_id] = len(chosen)
        for pair_no, (a, b) in enumerate(chosen, start=1):
            material = f"{seed}|ru-live|{model_id}|{pair_no}|{a['question_id']}|{b['question_id']}"
            rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16], 16))
            if rng.random() < 0.5:
                left, right = a, b
            else:
                left, right = b, a
            pair_id = hashlib.sha256(material.encode()).hexdigest()[:16]
            blind_rows.append([pair_id, left["sentence"], right["sentence"], ""])
            expected = "A" if left["live_score"] > right["live_score"] else "B"
            key_rows.append([
                pair_id,
                model_id,
                left["question_id"], left["sentence_index"], left["live_score"],
                right["question_id"], right["sentence_index"], right["live_score"],
                abs(left["live_score"] - right["live_score"]), expected,
            ])

    _write_csv(
        out_dir / "russian_live_blind_pairs.csv",
        ["pair_id", "sentence_A", "sentence_B", "more_AI_A_B_or_tie"],
        blind_rows,
    )
    _write_csv(
        out_dir / "russian_live_blind_pair_key.csv",
        [
            "pair_id","model_id",
            "qid_A","sentence_index_A","live_score_A",
            "qid_B","sentence_index_B","live_score_B",
            "live_score_gap","model_predicts_more_AI",
        ],
        key_rows,
    )

    # All complete Russian guided/random correction pairs, plus a deterministic
    # small high-signal subset (up to five/model) selected only by the PRE-TREATMENT
    # guided-target vs random-target LIVE score gap.
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT model_id,question_id,arm,target_live_score,corrected_text,status
            FROM correction_trials
            WHERE status='complete'
            ORDER BY model_id,question_id,arm
            """
        ).fetchall()
    trial = {
        (str(r["model_id"]), str(r["question_id"]), str(r["arm"])): r
        for r in rows
        if str(r["question_id"]) in qset and str(r["model_id"]) in set(models)
    }

    all_corr: list[list[Any]] = []
    all_key: list[list[Any]] = []
    small_corr: list[list[Any]] = []
    small_key: list[list[Any]] = []
    for model_id in models:
        available = []
        for qid in russian_qids:
            g = trial.get((model_id, qid, "guided"))
            r = trial.get((model_id, qid, "random"))
            if g is None or r is None:
                continue
            gap = float(g["target_live_score"]) - float(r["target_live_score"])
            available.append((gap, qid, g, r))
        available.sort(key=lambda x: (-x[0], x[1]))
        selected_small = {qid for _, qid, _, _ in available[:5]}
        for gap, qid, g, r in available:
            material = f"{seed}|ru-correction|{model_id}|{qid}"
            rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16], 16))
            if rng.random() < 0.5:
                arm_a, row_a, arm_b, row_b = "guided", g, "random", r
            else:
                arm_a, row_a, arm_b, row_b = "random", r, "guided", g
            pair_id = hashlib.sha256(material.encode()).hexdigest()[:16]
            public = [pair_id, str(row_a["corrected_text"]), str(row_b["corrected_text"]), ""]
            secret = [
                pair_id, model_id, qid, arm_a, arm_b,
                g["target_live_score"], r["target_live_score"], gap,
            ]
            all_corr.append(public)
            all_key.append(secret)
            if qid in selected_small:
                small_corr.append(public)
                small_key.append(secret)

    corr_header = ["pair_id","text_A","text_B","more_human_A_B_or_tie"]
    key_header = [
        "pair_id","model_id","question_id","arm_A","arm_B",
        "guided_target_live_score","random_target_live_score",
        "pre_treatment_target_score_gap",
    ]
    _write_csv(out_dir / "russian_correction_blind_pairs_all.csv", corr_header, all_corr)
    _write_csv(out_dir / "russian_correction_blind_pair_key_all.csv", key_header, all_key)
    _write_csv(out_dir / "russian_correction_blind_pairs_small.csv", corr_header, small_corr)
    _write_csv(out_dir / "russian_correction_blind_pair_key_small.csv", key_header, small_key)

    summary = {
        "live_blind_pairs_total": len(blind_rows),
        "live_blind_pairs_by_model": pair_counts,
        "correction_pairs_all": len(all_corr),
        "correction_pairs_small": len(small_corr),
        "selection_note": (
            "LIVE ranking pairs maximize pre-existing LIVE-score separation. "
            "Small correction subset selects largest pre-treatment guided-vs-random "
            "target-score gaps; it is a high-signal validation subset, not an unbiased "
            "estimate of average correction effect."
        ),
    }
    (out_dir / "russian_human_validation_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def export_validation_status(
    out_dir: Path,
    store: Store,
    *,
    russian_qids: Sequence[str],
    models: Sequence[str],
) -> None:
    qset = set(russian_qids)
    rows_out: list[list[Any]] = []
    with store.connect() as con:
        for model_id in models:
            cells = con.execute(
                """
                SELECT question_id,live_status,post_status,live_error,post_error
                FROM cells WHERE model_id=? ORDER BY question_id
                """,
                (model_id,),
            ).fetchall()
            for r in cells:
                if str(r["question_id"]) in qset:
                    rows_out.append([
                        model_id, r["question_id"], r["live_status"], r["post_status"],
                        r["live_error"], r["post_error"],
                    ])
    _write_csv(
        out_dir / "russian_validation_status.csv",
        ["model_id","question_id","live_status","post_status","live_error","post_error"],
        rows_out,
    )

def build_adaptive_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adaptive V5 AI-likeness live/post/replay experiment"
    )
    p.add_argument("--seed", type=int, default=20260907,
                   help="use a fresh seed; pilot/smoke data should not be reused")
    p.add_argument("--min-cases", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-cases", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("self_confidence_v5_adaptive"))
    p.add_argument(
        "--prompt-bank", type=Path, default=Path("prompts_v3.jsonl"),
        help="frozen v3 JSONL prompt bank (100 English primary + 10 Russian validation)",
    )
    p.add_argument(
        "--prompt-bank-sha", default=PROMPT_BANK_EXPECTED_SHA256,
        help="required SHA256 for the frozen prompt bank",
    )
    p.add_argument("--replay-fraction", type=float, default=0.10)
    p.add_argument("--models", default="all")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--per-model-workers", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=3,
                   help="transient network/429/5xx attempts; content_filter is never retried")
    p.add_argument("--max-retry-delay", type=float, default=20.0)
    p.add_argument("--live-max-tokens", type=int, default=4096)
    p.add_argument("--post-max-tokens", type=int, default=4096)
    p.add_argument("--replay-max-tokens", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--primary-model", default=DEFAULT_PRIMARY_MODEL)
    p.add_argument("--state-model", default=DEFAULT_STATE_MODEL)
    p.add_argument("--rho0", type=float, default=0.20,
                   help="minimum useful primary question-level Spearman association")
    p.add_argument("--min-state-shift", type=float, default=5.0,
                   help="minimum meaningful POST-LIVE shift for the prespecified state model")
    p.add_argument("--min-model-difference", type=float, default=5.0,
                   help="minimum meaningful difference in state shift between state and primary models")
    p.add_argument(
        "--primary-endpoint",
        choices=("live-post", "external-human"),
        default="live-post",
        help=(
            "live-post is fully automatic internal consistency; external-human "
            "uses --external-ratings and is the stronger external-validity endpoint"
        ),
    )
    p.add_argument("--external-ratings", type=Path, default=None,
                   help="CSV columns: model_id,question_id,human_score; sentence_index is optional")
    p.add_argument("--force-continue", action="store_true",
                   help="continue even if this output directory already contains a success decision")

    # V5.6 post-confirmatory extension mode. It operates in-place on an existing
    # V5.5 workspace and never regenerates successful baseline LIVE cells.
    p.add_argument("--extend-existing", action="store_true",
                   help="run V5.6 expanded replay + correction on an existing --out workspace")
    p.add_argument(
        "--extension-models",
        default="gpt-5.5,gemini-3.1-pro-preview",
        help="comma-separated frozen models for replay/correction extension",
    )
    p.add_argument("--expanded-replay", choices=("none","all"), default="all")
    p.add_argument("--self-correction", choices=("none","paired"), default="paired")
    p.add_argument("--repair-minor", action="store_true",
                   help="reparse stored harmless syntax failures and retry only small provider failures")
    p.add_argument("--extension-workers", type=int, default=16)
    p.add_argument("--extension-per-model-workers", type=int, default=3)
    p.add_argument("--correction-seed", type=int, default=2026090601)
    p.add_argument("--no-correction-post-assess", action="store_true",
                   help="generate corrected texts but skip same-model post diagnostic")

    # V5.7 fixed Russian native-language validation. It never contributes to
    # the English adaptive stopping decision.
    p.add_argument(
        "--validation-models",
        default="gpt-5.5,gemini-3.1-pro-preview",
        help="models used for the fixed 10-prompt Russian validation subset",
    )
    p.add_argument("--skip-russian-validation", action="store_true")
    p.add_argument("--skip-russian-replay", action="store_true")
    p.add_argument("--skip-russian-correction", action="store_true")
    p.add_argument(
        "--human-pair-min-gap", type=float, default=25.0,
        help="preferred minimum LIVE-score gap for Russian blind sentence pairs",
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def adaptive_main() -> int:
    parser = build_adaptive_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    print(
        f"Self-confidence adaptive runner V{CODE_VERSION} | prompt={PROMPT_VERSION} | "
        f"schema={SCHEMA_VERSION} | adaptive={ADAPTIVE_VERSION} | bank={PROMPT_BANK_VERSION}",
        flush=True,
    )
    print(f"Script: {script_path.name} | SHA256={script_sha}", flush=True)

    if args.self_test:
        adaptive_self_test()
        extension_self_test()
        # Bank/order self-test independent of any network call.
        if args.prompt_bank.exists():
            rows, sha = load_fixed_prompt_bank(args.prompt_bank, args.prompt_bank_sha)
            primary_test = balanced_primary_questions(rows, seed=args.seed)
            ru_test = russian_validation_questions(rows, seed=args.seed)
            assert len(primary_test) >= 1
            print(
                f"V5.7 prompt-bank self-test: primary={len(primary_test)} "
                f"russian={len(ru_test)} sha={sha}",
                flush=True,
            )
        else:
            print(
                "V5.7 prompt-bank self-test skipped because --prompt-bank does not exist; "
                "parser/extension tests passed.",
                flush=True,
            )
        return 0

    # Preserve the V5.6 in-place extension path for old workspaces.
    if args.extend_existing:
        return extension_main(
            args, parser, script_path=script_path, script_sha=script_sha
        )

    if not 0 < args.alpha < 1:
        parser.error("--alpha must be in (0,1)")
    if not -0.99 < args.rho0 < 0.99:
        parser.error("--rho0 must be between -0.99 and 0.99")
    if not 0 <= args.replay_fraction <= 1:
        parser.error("--replay-fraction must be in [0,1]")
    if args.human_pair_min_gap < 0:
        parser.error("--human-pair-min-gap must be >= 0")

    try:
        bank_rows, bank_sha = load_fixed_prompt_bank(
            args.prompt_bank, args.prompt_bank_sha
        )
        primary_bank = balanced_primary_questions(bank_rows, seed=args.seed)
        russian_bank = russian_validation_questions(bank_rows, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    primary_domains = sorted({str(q.metadata["domain"]) for q in primary_bank})
    n_domains = len(primary_domains)
    if n_domains < 2:
        parser.error("primary prompt bank must span multiple domains")
    if args.max_cases > len(primary_bank):
        parser.error(
            f"--max-cases={args.max_cases} exceeds frozen English primary bank "
            f"size {len(primary_bank)}"
        )
    if args.batch_size % n_domains != 0:
        parser.error(
            f"for domain-balanced looks, --batch-size must be a multiple of the "
            f"{n_domains} primary domains (recommended: {n_domains})"
        )
    try:
        looks = _planned_looks(args.min_cases, args.max_cases, args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        models = model_selection(args.models)
        validation_models = model_selection(args.validation_models)
        extension_models = model_selection(args.extension_models)
    except ValueError as exc:
        parser.error(str(exc))
    for needed in (args.primary_model, args.state_model):
        if needed not in models:
            parser.error(f"adaptive stopping requires model {needed!r} to be selected")
    for model_id in extension_models:
        if model_id not in models:
            parser.error(
                f"post-stop extension model {model_id!r} must also be selected in --models"
            )
    if args.primary_endpoint == "external-human" and args.external_ratings is None:
        parser.error("--primary-endpoint external-human requires --external-ratings PATH")

    all_questions = primary_bank[: args.max_cases]
    all_replay_ids = set(
        select_replay_questions_batched(
            all_questions,
            fraction=args.replay_fraction,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    )
    alpha_boundary = args.alpha / len(looks)
    bank_info = prompt_bank_summary(bank_rows, bank_sha)

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_bank_path = args.out / "prompt_bank_v3_frozen.jsonl"
    bank_bytes = args.prompt_bank.read_bytes()
    if frozen_bank_path.exists() and frozen_bank_path.read_bytes() != bank_bytes:
        parser.error(
            f"{frozen_bank_path} differs from --prompt-bank; use a fresh --out directory"
        )
    frozen_bank_path.write_bytes(bank_bytes)

    manifest_path = write_adaptive_manifest(
        args.out,
        args=args,
        questions=all_questions,
        models=models,
        replay_ids=sorted(all_replay_ids),
        planned_looks=looks,
        prompt_bank_info=bank_info,
        russian_questions=russian_bank,
    )

    print(f"Prompt bank: {args.prompt_bank} | SHA256={bank_sha}", flush=True)
    print(
        f"Frozen bank: English primary={len(primary_bank)} across {n_domains} domains; "
        f"Russian validation={len(russian_bank)}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(
        f"Adaptive English plan: start={args.min_cases}, batch={args.batch_size}, "
        f"hard_cap={args.max_cases}, planned_looks={looks}",
        flush=True,
    )
    print(
        f"Sequential rule: alpha={args.alpha:.4g}, {len(looks)} planned looks, "
        f"per-look Bonferroni boundary p<={alpha_boundary:.6g}",
        flush=True,
    )
    print(
        f"Primary: {args.primary_model} {args.primary_endpoint}, H0 rho<={args.rho0:.2f}",
        flush=True,
    )
    print(
        f"State distortion: {args.state_model} POST-LIVE > {args.min_state_shift:g}; "
        f"model dependence contrast > {args.min_model_difference:g}",
        flush=True,
    )
    print(
        f"Post-stop English extension models: {', '.join(extension_models)} | "
        f"full replay={args.expanded_replay == 'all'} | correction={args.self_correction == 'paired'}",
        flush=True,
    )
    if args.skip_russian_validation:
        print("Russian validation: SKIPPED by flag", flush=True)
    else:
        print(
            f"Russian validation: fixed 10 prompts, models={', '.join(validation_models)}, "
            f"full replay={not args.skip_russian_replay}, "
            f"correction={not args.skip_russian_correction}; NEVER used for stopping",
            flush=True,
        )
    print(
        "Retry policy: content_filter/parse failures fail immediately; transient "
        f"HTTP/network errors <= {args.retries} attempts, retry sleep <= {args.max_retry_delay:g}s",
        flush=True,
    )
    print(
        f"English adaptive replay: {len(all_replay_ids)} questions at hard cap "
        f"({args.replay_fraction:.1%}, sampled within balanced batches)",
        flush=True,
    )
    if args.primary_endpoint == "live-post":
        print(
            "NOTE: automatic stopping establishes INTERNAL live-vs-post self-assessment "
            "consistency. Human/Russian forced-choice files are secondary external-style validation.",
            flush=True,
        )

    if args.dry_run:
        for start in range(0, len(all_questions), args.batch_size):
            batch = all_questions[start:start + args.batch_size]
            counts: dict[str, int] = {}
            for q in batch:
                d = str(q.metadata["domain"])
                counts[d] = counts.get(d, 0) + 1
            print(
                f"  dry batch {start // args.batch_size + 1}: "
                + ", ".join(f"{d}={counts.get(d,0)}" for d in sorted(counts)),
                flush=True,
            )
        print("DRY RUN: no API calls made; SQLite DB not created.", flush=True)
        return 0

    load_simple_dotenv(Path.cwd() / ".env")
    load_simple_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is not set")

    db_path = args.out / "experiment.sqlite3"
    store = Store(db_path)
    _adaptive_schema(store)
    _extension_schema(store)
    store.write_meta("code_version", CODE_VERSION)
    store.write_meta("script_sha256", script_sha)
    store.write_meta("schema_version", SCHEMA_VERSION)
    store.write_meta("prompt_version", PROMPT_VERSION)
    store.write_meta("adaptive_version", ADAPTIVE_VERSION)
    store.write_meta("prompt_bank_version", PROMPT_BANK_VERSION)
    store.write_meta("prompt_bank_sha256", bank_sha)
    store.write_meta("primary_subset", PRIMARY_SUBSET)
    store.write_meta("russian_validation_subset", RUSSIAN_VALIDATION_SUBSET)
    store.write_meta("seed", args.seed)
    store.write_meta("min_cases", args.min_cases)
    store.write_meta("batch_size", args.batch_size)
    store.write_meta("max_cases", args.max_cases)
    store.write_meta("planned_looks", looks)
    store.write_meta("alpha", args.alpha)
    store.write_meta("alpha_boundary", alpha_boundary)
    store.write_meta("primary_endpoint", args.primary_endpoint)
    store.write_meta("primary_model", args.primary_model)
    store.write_meta("state_model", args.state_model)
    store.write_meta("rho0", args.rho0)
    store.write_meta("min_state_shift", args.min_state_shift)
    store.write_meta("min_model_difference", args.min_model_difference)
    store.write_meta("models", models)
    store.write_meta("validation_models", validation_models)
    store.write_meta("extension_models_v57", extension_models)

    client = FastFailOpenRouterClient(
        api_key=api_key,
        temperature=args.temperature,
        timeout_s=args.timeout,
        retries=args.retries,
        max_retry_delay_s=args.max_retry_delay,
    )

    # Resume semantics: an existing SUCCESS/HARD_LIMIT freezes the English sample
    # size but does NOT prevent unfinished post-stop replay/correction/Russian work
    # from resuming.
    with store.connect() as con:
        latest = con.execute(
            "SELECT decision,look_cases FROM adaptive_looks ORDER BY look_cases DESC LIMIT 1"
        ).fetchone()
    final_n: int | None = None
    if latest is not None and latest["decision"] in {"success", "hard_limit"}:
        final_n = int(latest["look_cases"])
        print(
            f"Existing English adaptive decision is frozen at N={final_n}: "
            f"{str(latest['decision']).upper()}. Baseline collection will not continue.",
            flush=True,
        )

    awaiting_ratings = False
    if final_n is None:
        look_number_by_n = {n: i + 1 for i, n in enumerate(looks)}
        for batch_start in range(0, args.max_cases, args.batch_size):
            batch = all_questions[batch_start:batch_start + args.batch_size]
            if not batch:
                break
            cumulative = batch_start + len(batch)
            batch_replay = {q.question_id for q in batch if q.question_id in all_replay_ids}

            print(
                f"\n=== ENGLISH ADAPTIVE BATCH {batch_start // args.batch_size + 1}: "
                f"questions {batch_start + 1}-{cumulative} ===",
                flush=True,
            )
            store.register(batch, models)
            _persist_replay_selection_incremental(
                store,
                [q.question_id for q in batch],
                batch_replay,
                args.replay_fraction,
                args.seed,
            )
            runner = AdaptiveRunner(
                store=store,
                client=client,
                questions=batch,
                models=models,
                workers=args.workers,
                per_model_workers=args.per_model_workers,
                live_max_tokens=args.live_max_tokens,
                post_max_tokens=args.post_max_tokens,
                replay_max_tokens=args.replay_max_tokens,
            )
            runner.run_live()
            runner.run_post()
            runner.run_replay_selected(batch_replay)
            export_results(args.out, store)
            current_qids = [q.question_id for q in all_questions[:cumulative]]
            write_external_ratings_template(
                args.out, store, args.primary_model, current_qids
            )

            if cumulative not in look_number_by_n:
                continue

            hard_limit = cumulative >= args.max_cases
            result = evaluate_adaptive_look(
                store=store,
                cases_attempted=cumulative,
                look_index=look_number_by_n[cumulative],
                alpha_boundary=alpha_boundary,
                primary_model=args.primary_model,
                state_model=args.state_model,
                primary_endpoint=args.primary_endpoint,
                external_ratings=args.external_ratings,
                rho0=args.rho0,
                min_state_shift=args.min_state_shift,
                min_model_difference=args.min_model_difference,
                hard_limit=hard_limit,
                question_ids=current_qids,
            )
            record_adaptive_look(store, result)
            export_adaptive_looks(args.out, store)

            pstat = result["primary"]
            sstat = result["state_distortion"]
            mstat = result["model_dependence"]
            print(
                f"[adaptive look N={cumulative}] boundary={alpha_boundary:.6g}\n"
                f"  primary: n={pstat['n_questions']} rho={pstat['spearman']} "
                f"p={pstat['p_one_sided']} passed={pstat['passed']}\n"
                f"  state: n={sstat['n_non_ties']} mean_shift={sstat['mean_post_minus_live']:.3f} "
                f"p={sstat['p_sign_one_sided']:.6g} passed={sstat['passed']}\n"
                f"  model-dependence: n={mstat['n_non_ties']} "
                f"mean_contrast={mstat['mean_contrast']:.3f} "
                f"p={mstat['p_sign_one_sided']:.6g} passed={mstat['passed']}\n"
                f"  DECISION: {result['decision'].upper()}",
                flush=True,
            )

            if result["decision"] == "awaiting_ratings":
                print(
                    "External-human mode is waiting for complete independent ratings at "
                    f"this English look. Fill {args.out / 'external_ratings_template.csv'} "
                    "and rerun the SAME command; completed calls will be skipped.",
                    flush=True,
                )
                awaiting_ratings = True
                break
            if result["decision"] in {"success", "hard_limit"}:
                final_n = cumulative
                break

    if awaiting_ratings:
        export_results(args.out, store)
        export_adaptive_looks(args.out, store)
        return 0

    if final_n is None:
        # Defensive fallback; with a valid plan max_cases is always a planned look.
        final_n = args.max_cases

    attempted_primary = all_questions[:final_n]
    attempted_primary_ids = [q.question_id for q in attempted_primary]
    write_external_ratings_template(
        args.out, store, args.primary_model, attempted_primary_ids
    )

    # Conservative repair of genuine provider failures / harmless stored syntax.
    if args.repair_minor:
        qmap_registered = {q.question_id: q for q in attempted_primary}
        reparsed, repaired_live = _repair_stored_parse_failures(store)
        retried = _retry_minor_provider_failures(
            store=store,
            client=client,
            questions=qmap_registered,
            routes=models,
            repaired_live=repaired_live,
            live_max_tokens=args.live_max_tokens,
            post_max_tokens=args.post_max_tokens,
        )
        print(
            f"[English minor repair] stored reparses={reparsed}; provider/unlocked retries={retried}",
            flush=True,
        )

    # Post-stop mechanistic/correction extension on the frozen English sample.
    ext_runner = ExtensionRunner(
        store=store,
        client=client,
        questions=attempted_primary,
        models=extension_models,
        workers=args.extension_workers,
        per_model_workers=args.extension_per_model_workers,
        live_max_tokens=args.live_max_tokens,
        post_max_tokens=args.post_max_tokens,
        replay_max_tokens=args.replay_max_tokens,
    )
    if args.expanded_replay == "all":
        print(f"\n=== POST-STOP ENGLISH FULL REPLAY: N={final_n} ===", flush=True)
        ext_runner.run_full_replay(attempted_primary_ids)
    if args.self_correction == "paired":
        print(f"\n=== POST-STOP ENGLISH PAIRED CORRECTION: N={final_n} ===", flush=True)
        ext_runner.run_corrections(attempted_primary_ids, args.correction_seed)
        if not args.no_correction_post_assess:
            ext_runner.run_correction_assessments()

    # Fixed Russian native-language secondary subset. It is deliberately run only
    # after the English stopping point is frozen and can never alter adaptive_looks.
    if not args.skip_russian_validation and russian_bank:
        print("\n=== FIXED RUSSIAN VALIDATION SUBSET: 10 PROMPTS ===", flush=True)
        store.register(russian_bank, validation_models)
        ru_runner = AdaptiveRunner(
            store=store,
            client=client,
            questions=russian_bank,
            models=validation_models,
            workers=args.extension_workers,
            per_model_workers=args.extension_per_model_workers,
            live_max_tokens=args.live_max_tokens,
            post_max_tokens=args.post_max_tokens,
            replay_max_tokens=args.replay_max_tokens,
        )
        ru_runner.run_live()
        ru_runner.run_post()

        ru_ext = ExtensionRunner(
            store=store,
            client=client,
            questions=russian_bank,
            models=validation_models,
            workers=args.extension_workers,
            per_model_workers=args.extension_per_model_workers,
            live_max_tokens=args.live_max_tokens,
            post_max_tokens=args.post_max_tokens,
            replay_max_tokens=args.replay_max_tokens,
        )
        ru_ids = [q.question_id for q in russian_bank]
        if not args.skip_russian_replay:
            ru_ext.run_full_replay(ru_ids)
        if not args.skip_russian_correction:
            ru_ext.run_corrections(ru_ids, args.correction_seed + 700001)
            if not args.no_correction_post_assess:
                ru_ext.run_correction_assessments()

        export_validation_status(
            args.out,
            store,
            russian_qids=ru_ids,
            models=list(validation_models),
        )
        human_summary = export_russian_human_validation(
            args.out,
            store,
            russian_qids=ru_ids,
            models=list(validation_models),
            seed=args.seed + 31337,
            preferred_min_gap=args.human_pair_min_gap,
        )
        print(
            "Russian blind human task exported: "
            f"live pairs={human_summary['live_blind_pairs_total']}, "
            f"small correction pairs={human_summary['correction_pairs_small']}",
            flush=True,
        )

    # Final combined exports. Adaptive look files remain English-only by design.
    export_results(args.out, store)
    export_adaptive_looks(args.out, store)
    _export_replay_extension(args.out, store, list(dict.fromkeys(
        list(extension_models) + ([] if args.skip_russian_validation else list(validation_models))
    )))
    _export_correction_extension(
        args.out,
        store,
        list(dict.fromkeys(
            list(extension_models) + ([] if args.skip_russian_validation else list(validation_models))
        )),
        args.correction_seed + 17,
    )

    run_summary = {
        "code_version": CODE_VERSION,
        "prompt_bank_version": PROMPT_BANK_VERSION,
        "prompt_bank_sha256": bank_sha,
        "english_adaptive_final_n": final_n,
        "english_primary_ids": attempted_primary_ids,
        "russian_validation_ids": [q.question_id for q in russian_bank]
        if not args.skip_russian_validation else [],
        "russian_validation_models": list(validation_models)
        if not args.skip_russian_validation else [],
        "english_extension_models": list(extension_models),
        "sentence_count_contract": [CASE_SENTENCE_MIN, CASE_SENTENCE_MAX],
    }
    (args.out / "run_summary_v57.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nV5.7 experiment finished.", flush=True)
    print(f"SQLite: {db_path}", flush=True)
    print(f"English adaptive looks: {args.out / 'adaptive_looks.csv'}", flush=True)
    print(f"English final N: {final_n}", flush=True)
    print(f"Prompt bank frozen copy: {frozen_bank_path}", flush=True)
    if not args.skip_russian_validation:
        print(f"Russian blind LIVE pairs: {args.out / 'russian_live_blind_pairs.csv'}", flush=True)
        print(
            f"Russian small correction pairs: {args.out / 'russian_correction_blind_pairs_small.csv'}",
            flush=True,
        )
        print("Do not inspect the corresponding *_key.csv files before rating.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(adaptive_main())
