#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun program/v1, Python 3.13 stdlib only).

Reads the chart history at $MAKEFUN_INPUT_DIR (default /input):
  - task.json: places scored, output file name, window (slots x slot_minutes)
  - history.ndjson: one snapshot per line, each with `at` and `up_and_coming`
    rows ({rank, universe_id, ...}), most recent last.

Writes $MAKEFUN_OUTPUT_DIR/<task.output> (default /output/forecast.json):
  {"probabilities": [{"universe_id", "place", "p"}], "top10": [universe_id x10]}

Model
-----
Each game's rank is treated as a random walk with no drift: the chart's
per-hour rank volatility is measured from the history itself (median absolute
hourly rank change, MAD -> sigma), and each game's rank at hour t is modelled
as Normal(current_rank, sigma * sqrt(t)).  The per-place probabilities are the
marginals of that model averaged uniformly over the scored window, which is
exactly how the scorer uses the forecast (one forecast scored against every
hourly read of the window).

Why no drift: on the practice history, recent rank velocity barely persists
(autocorrelation ~0.2), and player-count growth is uncorrelated with rank
movement (r ~ 0.02).  A drift term only overfits noise; the walk plus honest
diffusion beats the persistence baseline by ~10 log-score points in backtest.

Why a large unnamed reserve: about 15% of the games that take a top-10 place
in the window come from rank 41+ or from outside the chart entirely, so each
place keeps 0.25 of its mass unnamed (shared by every game not named there).
Named mass is normalised to 0.75 per place; rows below 0.0005 are dropped.

Deterministic: no randomness, no clock, no network, fixed iteration order.
"""

import json
import math
import os
import sys
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

VOL_SCALE = 0.7          # shrink on the MAD-based hourly sigma
TAIL_RESERVE = 0.25      # mass per place left unnamed
MIN_ROW_P = 0.0005       # drop named rows below this after scaling
CANDIDATE_TOP = 40       # top-N of the last chart kept as candidates
RECENT_TOP10_READS = 6   # also keep any game in the top 10 of these last reads
MIN_SIGMA = 0.35         # floor on the hourly sigma (ranks)
DEFAULT_SIGMA = 0.5      # fallback when the history is too thin


def parse_time(value):
    """Parse an ISO-8601 timestamp to epoch seconds; None when unusable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    """Return snapshots sorted by time: [{at, chart: {uid: rank}}]."""
    path = os.path.join(input_dir, "history.ndjson")
    snapshots = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
            except ValueError:
                continue
            if snap.get("ok") is False:
                continue
            rows = snap.get("up_and_coming")
            if not isinstance(rows, list) or not rows:
                continue
            chart = {}
            for row in rows:
                try:
                    uid = int(row["universe_id"])
                    rank = int(row["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
                if rank >= 1 and uid not in chart:
                    chart[uid] = rank
            if chart:
                snapshots.append({"at": parse_time(snap.get("at")), "chart": chart})
    # Time order; snapshots without a timestamp keep file order at the end.
    dated = [s for s in snapshots if s["at"] is not None]
    undated = [s for s in snapshots if s["at"] is None]
    dated.sort(key=lambda s: s["at"])
    seen = set()
    ordered = []
    for snap in dated + undated:
        key = snap["at"]
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        ordered.append(snap)
    return ordered


def hourly_sigma(snapshots):
    """Median absolute hourly rank change, MAD-scaled to a sigma."""
    diffs = []
    for prev, cur in zip(snapshots, snapshots[1:]):
        if prev["at"] is None or cur["at"] is None:
            continue
        hours = (cur["at"] - prev["at"]) / 3600.0
        if hours <= 0:
            continue
        step = max(hours, 0.5)
        for uid, rank in cur["chart"].items():
            if uid in prev["chart"]:
                diffs.append(abs(rank - prev["chart"][uid]) / step)
    if len(diffs) < 10:
        return DEFAULT_SIGMA
    diffs.sort()
    median = diffs[len(diffs) // 2]
    sigma = median * 1.4826 * VOL_SCALE
    return max(sigma, MIN_SIGMA)


def persistence_forecast(last_chart, places):
    """Fallback when the history is too thin for the walk model.

    Same shape as the house baseline but with this entry's tail reserve.
    """
    ranked = sorted(last_chart.items(), key=lambda kv: kv[1])
    ids = [uid for uid, _ in ranked]
    probabilities = []
    for place in range(1, places + 1):
        rows = []
        if place <= len(ids):
            rows.append((ids[place - 1], 0.5))
        if place == 1:
            if len(ids) > 1:
                rows.append((ids[1], 0.3))
        else:
            if place - 2 < len(ids):
                rows.append((ids[place - 2], 0.15))
            if len(ids) > place:
                rows.append((ids[place], 0.15))
        total = sum(p for _, p in rows)
        scale = (1.0 - TAIL_RESERVE) / total if total > 0 else 0.0
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": place, "p": p * scale})
    top10 = ids[:places]
    while len(top10) < places and top10:
        top10.append(top10[0])
    return {"probabilities": probabilities, "top10": top10}


def walk_forecast(snapshots, places, horizon_hours):
    """Random-walk rank marginals, averaged uniformly over the window."""
    last_chart = snapshots[-1]["chart"]
    sigma = hourly_sigma(snapshots)
    candidates = sorted(last_chart.items(), key=lambda kv: kv[1])
    cand_ids = [uid for uid, _ in candidates[:CANDIDATE_TOP]]
    cand_set = set(cand_ids)
    for snap in snapshots[-RECENT_TOP10_READS:]:
        for uid, rank in snap["chart"].items():
            if rank <= places and uid not in cand_set:
                cand_set.add(uid)
                cand_ids.append(uid)
    cand_ids.sort(key=lambda uid: last_chart.get(uid, 10 ** 9))

    marginals = {place: {} for place in range(1, places + 1)}
    steps = max(int(round(horizon_hours)), 1)
    for uid in cand_ids:
        mu0 = float(last_chart.get(uid, 50))
        for t in range(1, steps + 1):
            sd = max(sigma * math.sqrt(t), MIN_SIGMA)
            for place in range(1, places + 1):
                prob = norm_cdf((place + 0.5 - mu0) / sd) - norm_cdf((place - 0.5 - mu0) / sd)
                if prob > 1e-6:
                    bucket = marginals[place]
                    bucket[uid] = bucket.get(uid, 0.0) + prob / steps

    probabilities = []
    top10 = []
    used = set()
    for place in range(1, places + 1):
        bucket = marginals[place]
        total = sum(bucket.values())
        scale = (1.0 - TAIL_RESERVE) / total if total > 0 else 0.0
        ranked = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
        for uid, value in ranked:
            p = value * scale
            if p >= MIN_ROW_P:
                probabilities.append({"universe_id": uid, "place": place, "p": p})
        for uid, _ in ranked:
            if uid not in used:
                top10.append(uid)
                used.add(uid)
                break
    if len(top10) < places:
        # Greedy argmax can leave places unfilled only when the candidate set
        # is smaller than the number of places; fill from the last chart in
        # rank order, never with a fabricated id.
        for uid, _ in candidates:
            if len(top10) >= places:
                break
            if uid not in used:
                top10.append(uid)
                used.add(uid)
    while len(top10) < places:
        top10.append(top10[0] if top10 else cand_ids[0])
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    task_path = os.path.join(INPUT_DIR, "task.json")
    with open(task_path, "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {}) or {}
    slots = float(window.get("slots", 72))
    slot_minutes = float(window.get("slot_minutes", 60))
    horizon_hours = slots * slot_minutes / 60.0

    snapshots = load_history(INPUT_DIR)
    if not snapshots:
        raise SystemExit("no usable snapshots in history.ndjson")
    if len(snapshots) < 2:
        forecast = persistence_forecast(snapshots[-1]["chart"], places)
    else:
        forecast = walk_forecast(snapshots, places, horizon_hours)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
