"""Analyze paired DMOSpeech handover results and prepare blind A/B listening packets.

Uses only the Python standard library. Run ``python analyze_results.py --help``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


CASE_FIELDS = ("sample_id", "speaker_id", "seed")
RESULT_FIELDS = (*CASE_FIELDS, "method")
METRICS = (
    ("wer", "WER (lower is better)"),
    ("speaker_similarity", "Speaker similarity (higher is better)"),
    ("inference_seconds", "Inference seconds (lower is better)"),
    ("rtf", "Real-time factor (lower is better)"),
    ("peak_flux", "Peak spectral flux (diagnostic only)"),
    ("mean_flux", "Mean spectral flux (diagnostic only)"),
)
AUDIT_FIELDS = ("wer", "speaker_similarity", "audio_seconds", "peak_flux", "mean_flux", "effective_w")


def load_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_results(path: Path) -> dict[tuple[str, str, str], dict[str, dict]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if not isinstance(source, list):
            raise ValueError("Legacy benchmark JSON must be a list of samples")
        rows = []
        for item in source:
            for method, evaluation in item.get("evaluations", {}).items():
                rows.append({
                    "sample_id": str(item["sample_id"]),
                    "speaker_id": str(item["speaker_id"]),
                    "seed": str(item.get("seed", "unspecified")),
                    "method": method,
                    "audio_file": evaluation.get("output_file", ""),
                    "gen_text": item.get("gen_text", ""),
                    **evaluation,
                })
    else:
        rows = load_csv(path, RESULT_FIELDS)
    grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    sample_speakers: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows, 1):
        if any(not str(row.get(field, "")).strip() for field in RESULT_FIELDS):
            raise ValueError(f"{path}: result row {index} has a blank case or method field")
        case = tuple(str(row[field]).strip() for field in CASE_FIELDS)
        method = str(row["method"]).strip()
        sample_seed = (case[0], case[2])
        old_speaker = sample_speakers.setdefault(sample_seed, case[1])
        if old_speaker != case[1]:
            raise ValueError(f"{path}: sample {case[0]}, seed {case[2]} has two speaker IDs")
        if method in grouped[case]:
            raise ValueError(f"{path}: duplicate case/method {case}/{method}")
        grouped[case][method] = row
    if not grouped:
        raise ValueError(f"{path}: no results found")
    return dict(grouped)


def number(row: dict, field: str) -> float | None:
    raw = row.get(field)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}={raw!r}")
    return value


def metric(row: dict, field: str) -> float | None:
    if field != "rtf":
        return number(row, field)
    seconds = number(row, "inference_seconds")
    audio_seconds = number(row, "audio_seconds")
    if seconds is None or audio_seconds is None or audio_seconds <= 0:
        return None
    return seconds / audio_seconds


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def cluster_summary(entries: list[tuple[str, float]], draws: int, seed: int) -> tuple[float, float | None, float | None, int]:
    """Equal weight per speaker; bootstrap resamples whole speakers."""
    groups: dict[str, list[float]] = defaultdict(list)
    for speaker, value in entries:
        groups[speaker].append(value)
    speaker_means = [sum(values) / len(values) for values in groups.values()]
    estimate = sum(speaker_means) / len(speaker_means)
    if len(groups) < 2:
        return estimate, None, None, len(groups)
    rng = random.Random(seed)
    replicates = [
        sum(rng.choices(speaker_means, k=len(speaker_means))) / len(speaker_means)
        for _ in range(draws)
    ]
    return estimate, percentile(replicates, 0.025), percentile(replicates, 0.975), len(groups)


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def audio_path(raw: str, results_path: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else results_path.parent / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_audit(grouped: dict, results_path: Path, baseline: str, copy: str) -> list[str]:
    pairs = [(case, methods[baseline], methods[copy]) for case, methods in grouped.items()
             if baseline in methods and copy in methods]
    if not pairs:
        return [f"Duplicate baseline audit: no matched `{copy}` branch is available."]
    available = sum(baseline in methods for methods in grouped.values())
    metric_mismatches = 0
    audio_equal = 0
    audio_different = 0
    audio_missing = 0
    for _, left, right in pairs:
        for field in AUDIT_FIELDS:
            a, b = left.get(field), right.get(field)
            if a is None or b is None or str(a).strip() == "" or str(b).strip() == "":
                continue
            if str(a) != str(b):
                metric_mismatches += 1
        a_path, b_path = left.get("audio_file", ""), right.get("audio_file", "")
        if not a_path or not b_path:
            audio_missing += 1
            continue
        a_file, b_file = audio_path(str(a_path), results_path), audio_path(str(b_path), results_path)
        if not a_file.is_file() or not b_file.is_file():
            audio_missing += 1
        elif sha256_file(a_file) == sha256_file(b_file):
            audio_equal += 1
        else:
            audio_different += 1
    status = "PASS" if audio_different == 0 and audio_missing == 0 and metric_mismatches == 0 and len(pairs) == available else "INCOMPLETE OR FAIL"
    return [
        f"Duplicate baseline audit: **{status}**. Matched cases: {len(pairs)}/{available}; "
        f"byte-identical audio: {audio_equal}; different audio: {audio_different}; "
        f"missing/unreadable audio: {audio_missing}; metric field mismatches: {metric_mismatches}.",
        "Inference time is excluded from the equality audit because separate executions can have different runtime.",
    ]


def coverage(grouped: dict, manifest: Path | None, seeds: list[str] | None,
             required_methods: tuple[str, ...]) -> list[str]:
    incomplete = [(case, sorted(set(required_methods) - set(methods)))
                  for case, methods in grouped.items()
                  if set(required_methods) - set(methods)]
    if manifest is None:
        lines = ["Expected-case coverage was not checked; pass `--manifest` and `--seeds` to audit missing tasks."]
    else:
        if not seeds:
            raise ValueError("--manifest requires --seeds")
        with manifest.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if not isinstance(source, list):
            raise ValueError(f"{manifest}: expected a list of cases")
        expected = {(str(item["sample_id"]), str(item["speaker_id"]), str(seed))
                    for item in source for seed in seeds}
        if len(expected) != len(source) * len(seeds):
            raise ValueError(f"{manifest}: duplicate sample/speaker case in manifest")
        found = set(grouped)
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        lines = [
            f"Expected cases: **{len(expected)}** ({len(source)} manifest prompts x {len(seeds)} seeds). "
            f"Present: **{len(expected & found)}**; missing: **{len(missing)}**; "
            f"unexpected: **{len(unexpected)}**.",
        ]
        if missing:
            example = ", ".join(f"{case[0]}/{case[1]}/seed{case[2]}" for case in missing[:12])
            lines.append(f"Missing expected cases (first 12): {example}.")
    lines.append(f"Cases missing one or more required methods: **{len(incomplete)}**.")
    if incomplete:
        example = ", ".join(f"{case[0]}/seed{case[2]} missing {','.join(methods)}"
                            for case, methods in incomplete[:12])
        lines.append(f"Incomplete cases (first 12): {example}.")
    return lines


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for index in ordered[i:j]:
            result[index] = rank
        i = j
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return numerator / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None


def disagreement_associations(grouped: dict, args: argparse.Namespace) -> list[str]:
    if not any(number(methods[args.baseline], "disagreement") is not None
               for methods in grouped.values() if args.baseline in methods):
        return ["Normalized disagreement `rho = ||S-T||/(||T||+epsilon)` was not recorded."]
    outcome_specs = (
        ("Baseline WER", "wer", "baseline"),
        ("WER reduction from proposed", "wer", "gain"),
        ("Baseline speaker similarity", "speaker_similarity", "baseline"),
        ("Speaker similarity gain from proposed", "speaker_similarity", "gain"),
    )
    lines = [
        "Normalized disagreement is `rho = ||S-T||/(||T||+epsilon)`. Each coefficient below is "
        "the mean of within-speaker Spearman correlations across speakers with at least three "
        "non-tied cases. Positive correlations are only associations, not causal evidence.",
        "",
        "| Outcome | Eligible speakers | Mean within-speaker rho | 95% speaker bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for label, field, kind in outcome_specs:
        by_speaker = defaultdict(list)
        for case, methods in grouped.items():
            baseline = methods.get(args.baseline)
            proposed = methods.get(args.proposed)
            if baseline is None or (kind == "gain" and proposed is None):
                continue
            rho = number(baseline, "disagreement")
            base = metric(baseline, field)
            new = metric(proposed, field) if proposed is not None else None
            if rho is None or base is None or (kind == "gain" and new is None):
                continue
            value = base if kind == "baseline" else (base - new if field == "wer" else new - base)
            by_speaker[case[1]].append((rho, value))
        correlations = []
        for speaker, pairs in by_speaker.items():
            coefficient = spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            if coefficient is not None:
                correlations.append((speaker, coefficient))
        if correlations:
            estimate, lower, upper, n_speakers = cluster_summary(correlations, args.bootstrap, args.seed)
            interval = f"[{fmt(lower)}, {fmt(upper)}]" if lower is not None else "NA (<2 speakers)"
            lines.append(f"| {label} | {n_speakers} | {fmt(estimate)} | {interval} |")
        else:
            lines.append(f"| {label} | 0 | NA | NA |")
    return lines


def load_preferences(ratings_path: Path, key_path: Path) -> tuple[dict[tuple, list[tuple[str | None, str]]], int, int]:
    keys = load_csv(key_path, ("trial_id", *CASE_FIELDS, "method_a", "method_b"))
    lookup = {}
    for row in keys:
        trial = row["trial_id"].strip()
        if not trial or trial in lookup:
            raise ValueError(f"{key_path}: blank or repeated trial_id {trial!r}")
        lookup[trial] = row
    ratings = load_csv(ratings_path, ("listener_id", "trial_id", "choice"))
    seen = set()
    scores: dict[tuple, list[float]] = defaultdict(list)
    listeners = set()
    for index, row in enumerate(ratings, 2):
        choice = row["choice"].strip().upper()
        if not choice:
            continue
        listener, trial = row["listener_id"].strip(), row["trial_id"].strip()
        if not listener or trial not in lookup:
            raise ValueError(f"{ratings_path}:{index}: missing listener_id or unknown trial_id")
        if choice not in {"A", "B", "TIE"}:
            raise ValueError(f"{ratings_path}:{index}: choice must be A, B, or tie")
        if (listener, trial) in seen:
            raise ValueError(f"{ratings_path}:{index}: repeated listener/trial rating")
        seen.add((listener, trial))
        listeners.add(listener)
        key = lookup[trial]
        case = tuple(key[field].strip() for field in CASE_FIELDS)
        method_a, method_b = key["method_a"].strip(), key["method_b"].strip()
        winner = method_a if choice == "A" else method_b if choice == "B" else None
        pair = tuple(sorted((method_a, method_b)))
        # Keep the winner as a method; analysis chooses its own target method.
        scores[(case, pair)].append((winner, listener))
    return scores, len(seen), len(listeners)


def load_naturalness(ratings_path: Path, key_path: Path, baseline: str) -> tuple[dict[tuple, list[int]], int, int]:
    keys = load_csv(key_path, ("trial_id", *CASE_FIELDS, "method"))
    lookup = {}
    for row in keys:
        trial = row["trial_id"].strip()
        if not trial or trial in lookup or row["method"].strip() != baseline:
            raise ValueError(f"{key_path}: invalid or repeated baseline naturalness trial {trial!r}")
        lookup[trial] = row
    ratings = load_csv(ratings_path, ("listener_id", "trial_id", "rating"))
    seen = set()
    scores: dict[tuple, list[int]] = defaultdict(list)
    listeners = set()
    for index, row in enumerate(ratings, 2):
        raw = row["rating"].strip()
        if not raw:
            continue
        listener, trial = row["listener_id"].strip(), row["trial_id"].strip()
        if not listener or trial not in lookup:
            raise ValueError(f"{ratings_path}:{index}: missing listener_id or unknown trial_id")
        if raw not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{ratings_path}:{index}: naturalness rating must be an integer from 1 to 5")
        if (listener, trial) in seen:
            raise ValueError(f"{ratings_path}:{index}: repeated listener/trial rating")
        seen.add((listener, trial))
        listeners.add(listener)
        case = tuple(lookup[trial][field].strip() for field in CASE_FIELDS)
        scores[case].append(int(raw))
    return scores, len(seen), len(listeners)


def naturalness_report(grouped: dict, args: argparse.Namespace) -> list[str]:
    if not args.naturalness_ratings and not args.naturalness_key:
        return ["**No single-clip baseline naturalness ratings supplied. Baseline naturalness and "
                "its association with disagreement remain untested.**"]
    if not args.naturalness_ratings or not args.naturalness_key:
        raise ValueError("--naturalness-ratings and --naturalness-key must be provided together")
    ratings, count, listeners = load_naturalness(args.naturalness_ratings,
                                                 args.naturalness_key, args.baseline)
    for case in ratings:
        if case not in grouped or args.baseline not in grouped[case]:
            raise ValueError(f"naturalness trial has no matching baseline result: {case}")
    if not ratings:
        return ["**No completed single-clip naturalness ratings supplied.**"]
    entries = [(case[1], sum(values) / len(values)) for case, values in ratings.items()]
    estimate, lower, upper, speakers = cluster_summary(entries, args.bootstrap, args.seed)
    interval = f"[{fmt(lower)}, {fmt(upper)}]" if lower is not None else "NA (<2 speakers)"
    under_rated = sum(len(values) < 3 for values in ratings.values())
    lines = [
        f"Baseline naturalness: **{fmt(estimate)} / 5** (95% speaker bootstrap interval {interval}); "
        f"{len(ratings)} rated cases, {speakers} speakers, {count} ratings from {listeners} listeners.",
        f"Cases with fewer than three independent listener ratings: **{under_rated}**.",
        "The 1-5 scale rates naturalness without reference speech. It does not measure speaker fidelity.",
        "",
        "| Speaker | Rated cases | Distinct prompts | Ratings |",
        "|---|---:|---:|---:|",
    ]
    for speaker in sorted({case[1] for case in ratings}):
        speaker_cases = [(case, values) for case, values in ratings.items() if case[1] == speaker]
        lines.append(f"| {speaker} | {len(speaker_cases)} | "
                     f"{len({case[0] for case, _ in speaker_cases})} | "
                     f"{sum(len(values) for _, values in speaker_cases)} |")
    lines += ["", "Disagreement associations below are exploratory and require at least four "
              "distinct prompts per speaker. Speakers below that threshold are excluded."]
    by_speaker = defaultdict(list)
    for case, values in ratings.items():
        rho = number(grouped[case][args.baseline], "disagreement")
        if rho is not None:
            by_speaker[case[1]].append((case[0], rho, sum(values) / len(values)))
    correlations = []
    for speaker, pairs in by_speaker.items():
        if len({pair[0] for pair in pairs}) < 4:
            continue
        coefficient = spearman([pair[1] for pair in pairs], [pair[2] for pair in pairs])
        if coefficient is not None:
            correlations.append((speaker, coefficient))
    if correlations:
        association, assoc_lower, assoc_upper, eligible = cluster_summary(
            correlations, args.bootstrap, args.seed)
        assoc_interval = (f"[{fmt(assoc_lower)}, {fmt(assoc_upper)}]"
                          if assoc_lower is not None else "NA (<2 speakers)")
        lines.append(
            f"Within-speaker Spearman association of normalized disagreement with baseline "
            f"naturalness: **{fmt(association)}** (95% speaker bootstrap interval {assoc_interval}; "
            f"{eligible} eligible speakers). Negative values mean higher disagreement is associated "
            "with lower naturalness; this is not causal proof."
        )
    else:
        lines.append("Disagreement/naturalness association: NA (too few varying rated cases per speaker).")
    return lines


def preference_associations(preferences: dict, grouped: dict,
                            args: argparse.Namespace) -> list[str]:
    lines = [
        "",
        "Exploratory within-speaker association of normalized disagreement with preference for proposed. "
        "Positive values mean higher disagreement coincided with more preference for proposed. "
        "At least four distinct prompts per speaker are required; this association does not "
        "prove the handover caused a quality loss.",
        "",
        "| Comparison | Eligible speakers | Mean within-speaker Spearman rho | 95% speaker bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for comparator in (args.baseline, args.fixed):
        pair = tuple(sorted((args.proposed, comparator)))
        by_speaker = defaultdict(list)
        for (case, trial_pair), votes in preferences.items():
            if trial_pair != pair or case not in grouped or args.baseline not in grouped[case]:
                continue
            rho = number(grouped[case][args.baseline], "disagreement")
            if rho is None:
                continue
            score = sum(0.5 if winner is None else float(winner == args.proposed)
                        for winner, _ in votes) / len(votes)
            by_speaker[case[1]].append((case[0], rho, score))
        coefficients = []
        for speaker, pairs in by_speaker.items():
            if len({value[0] for value in pairs}) < 4:
                continue
            coefficient = spearman([value[1] for value in pairs],
                                   [value[2] for value in pairs])
            if coefficient is not None:
                coefficients.append((speaker, coefficient))
        if coefficients:
            estimate, lower, upper, n_speakers = cluster_summary(coefficients, args.bootstrap, args.seed)
            interval = f"[{fmt(lower)}, {fmt(upper)}]" if lower is not None else "NA (<2 speakers)"
            lines.append(f"| proposed vs {comparator} | {n_speakers} | {fmt(estimate)} | {interval} |")
        else:
            lines.append(f"| proposed vs {comparator} | 0 | NA | NA |")
    return lines


def analyze(args: argparse.Namespace) -> None:
    path = args.results.resolve()
    grouped = load_results(path)
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    methods = (args.baseline, args.fixed, args.proposed)
    if len(set(methods)) != 3:
        raise ValueError("baseline, fixed, and proposed method names must differ")
    speakers = {case[1] for case in grouped}
    lines = [
        "# Paired teacher-to-student handover evaluation",
        "",
        f"Input: `{path}`. Cases: **{len(grouped)}**; speakers: **{len(speakers)}**.",
        "One case is a sample ID, speaker ID, and random seed. All differences below pair the same case.",
        "Speaker summaries give equal weight to each speaker. The 95% intervals resample whole speakers "
        f"with replacement ({args.bootstrap} draws; random seed {args.seed}).",
        "",
        "## Causal control",
        "",
        *baseline_audit(grouped, path, args.baseline, args.baseline_copy),
        "",
        "## Expected-case coverage",
        "",
        *coverage(grouped, args.manifest, args.seeds,
                  (args.baseline, args.baseline_copy, args.fixed, args.proposed)),
        "",
        "## Objective outcomes",
        "",
        "Deltas are proposed minus comparator. For WER, time, and real-time factor, negative favors proposed; "
        "for speaker similarity, positive favors proposed. Flux is diagnostic and does not establish audible quality.",
        "",
        "Wins/losses/ties count cases where proposed is better/worse/equal; they are not statistical tests.",
        "",
        "| Metric | Comparison | Paired cases | Speakers | Wins/Losses/Ties | Speaker-mean delta | 95% speaker bootstrap interval |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if len(speakers) < 20:
        lines.insert(5, f"**Caution:** only {len(speakers)} speakers are represented. Speaker bootstrap intervals may be unstable; treat them as exploratory.")
    for field, label in METRICS:
        for comparator in (args.baseline, args.fixed):
            entries = []
            for case, values in grouped.items():
                if args.proposed not in values or comparator not in values:
                    continue
                a, b = metric(values[args.proposed], field), metric(values[comparator], field)
                if a is not None and b is not None:
                    entries.append((case[1], a - b))
            if entries:
                mean, lower, upper, n_speakers = cluster_summary(entries, args.bootstrap, args.seed)
                interval = f"[{fmt(lower)}, {fmt(upper)}]" if lower is not None else "NA (<2 speakers)"
                direction = 1 if field == "speaker_similarity" else -1
                wins = sum(delta * direction > 0 for _, delta in entries)
                losses = sum(delta * direction < 0 for _, delta in entries)
                ties = len(entries) - wins - losses
                lines.append(f"| {label} | {args.proposed} - {comparator} | {len(entries)} | {n_speakers} | {wins}/{losses}/{ties} | {fmt(mean)} | {interval} |")
            else:
                lines.append(f"| {label} | {args.proposed} - {comparator} | 0 | 0 | 0/0/0 | NA | NA |")
    weights = [number(values[args.proposed], "effective_w") for values in grouped.values() if args.proposed in values]
    weights = [weight for weight in weights if weight is not None]
    if weights:
        lines += ["", f"Proposed acceptance weight: mean {fmt(sum(weights) / len(weights))}, "
                  f"range [{fmt(min(weights))}, {fmt(max(weights))}] over {len(weights)} cases."]
    lines += ["", "## Disagreement diagnostic", "", *disagreement_associations(grouped, args),
              "", "## Blinded listener preference", ""]
    if args.ratings or args.key:
        if not args.ratings or not args.key:
            raise ValueError("--ratings and --key must be provided together")
        preferences, n_ratings, n_listeners = load_preferences(args.ratings, args.key)
        under_rated = sum(len(votes) < 3 for votes in preferences.values())
        lines += [
            f"Scored ratings: **{n_ratings}** from **{n_listeners}** listeners. "
            "A preference score is 1 when proposed is chosen, 0 when the comparator is chosen, "
            "and 0.5 for a tie. Scores are averaged per case, then per speaker.",
            f"Rated comparisons with fewer than three independent listener ratings: **{under_rated}**. "
            "Plan for at least three independent listeners per trial.",
            "",
            "| Comparison | Rated cases | Speakers | Ratings | Proposed preference | 95% speaker bootstrap interval |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for comparator in (args.baseline, args.fixed):
            pair = tuple(sorted((args.proposed, comparator)))
            entries = []
            n = 0
            for (case, trial_pair), votes in preferences.items():
                if trial_pair != pair:
                    continue
                if case not in grouped or args.proposed not in grouped[case] or comparator not in grouped[case]:
                    raise ValueError(f"rating trial has no matching paired result: {case}, {pair}")
                case_score = sum(0.5 if winner is None else float(winner == args.proposed)
                                 for winner, _ in votes) / len(votes)
                entries.append((case[1], case_score))
                n += len(votes)
            if entries:
                estimate, lower, upper, n_speakers = cluster_summary(entries, args.bootstrap, args.seed)
                interval = f"[{fmt(lower)}, {fmt(upper)}]" if lower is not None else "NA (<2 speakers)"
                lines.append(f"| {args.proposed} vs {comparator} | {len(entries)} | {n_speakers} | {n} | {fmt(estimate)} | {interval} |")
            else:
                lines.append(f"| {args.proposed} vs {comparator} | 0 | 0 | 0 | NA | NA |")
        if not n_ratings:
            lines += ["", "**No completed ratings: perceptual quality has not been evaluated.**"]
        else:
            lines += preference_associations(preferences, grouped, args)
    else:
        lines += ["**No pairwise listener ratings supplied. This report cannot establish whether the proposed rule improves perceptual quality.**"]
    lines += ["", "## Baseline naturalness", "", *naturalness_report(grouped, args)]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- These confidence intervals describe speaker resampling within this test set. They do not correct for "
        "tuning on test speakers, listener sampling, or dataset shift.",
        "- A claim that the proposed rule improves perceptual quality requires blinded ratings on a held-out set. "
        "A claim that adaptation itself helps requires improvement over the fixed rule.",
        "- WER is an utterance-level average here. It is not a corpus word error rate. "
        "Missing pairs are excluded separately for each metric; compare the paired case counts.",
        "- A byte-identical duplicate baseline is a useful control, but the metrics file alone cannot prove "
        "that later noise and all preprocessing were shared. Verify the generator's logs and configuration.",
        "",
    ]
    report = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(report)


def select_cases(grouped: dict, maximum: int | None, rng: random.Random) -> list[tuple[str, str, str]]:
    """Round-robin over speakers, spreading each speaker's cases across seeds."""
    if maximum is not None and maximum < 1:
        raise ValueError("--max-cases must be positive")
    if maximum is not None and maximum > len(grouped):
        raise ValueError(f"requested {maximum} cases, but only {len(grouped)} are available")
    strata = defaultdict(list)
    for case in grouped:
        strata[(case[1], case[2])].append(case)
    for bucket in strata.values():
        rng.shuffle(bucket)
    speakers = sorted({case[1] for case in grouped})
    rng.shuffle(speakers)
    seed_order = sorted({case[2] for case in grouped})
    rng.shuffle(seed_order)
    seed_rank = {seed: index for index, seed in enumerate(seed_order)}
    global_seed_counts = defaultdict(int)
    speaker_seed_counts = defaultdict(int)
    target = len(grouped) if maximum is None else maximum
    selected = []
    while len(selected) < target:
        for speaker in speakers:
            options = [seed for seed in seed_order if strata[(speaker, seed)]]
            if not options:
                continue
            seed = min(options, key=lambda item: (
                speaker_seed_counts[(speaker, item)], global_seed_counts[item], seed_rank[item]))
            selected.append(strata[(speaker, seed)].pop())
            speaker_seed_counts[(speaker, seed)] += 1
            global_seed_counts[seed] += 1
            if len(selected) == target:
                break
    return selected


def make_packet(args: argparse.Namespace) -> None:
    results_path = args.results.resolve()
    grouped = load_results(results_path)
    pairs = ((args.proposed, args.baseline), (args.proposed, args.fixed))
    if len({args.proposed, args.baseline, args.fixed}) != 3:
        raise ValueError("baseline, fixed, and proposed method names must differ")
    rng = random.Random(args.seed)
    selected = select_cases(grouped, args.max_cases, rng)
    naturalness_candidates = {
        case: methods for case, methods in grouped.items()
        if case[2] == args.naturalness_seed
        and all(method in methods for method in
                (args.baseline, args.baseline_copy, args.fixed, args.proposed))
    }
    if not naturalness_candidates:
        raise ValueError(f"no complete cases with naturalness synthesis seed {args.naturalness_seed}")
    naturalness_rng = random.Random(args.seed + 1)
    naturalness_cases = select_cases(naturalness_candidates, args.naturalness_max_cases,
                                     naturalness_rng)
    naturalness_sources = {}
    for case in naturalness_cases:
        raw = grouped[case][args.baseline].get("audio_file", "")
        if not raw:
            raise ValueError(f"naturalness case {case}: baseline needs audio_file")
        source = audio_path(str(raw), results_path)
        if not source.is_file():
            raise FileNotFoundError(f"naturalness case {case}: missing baseline audio {source}")
        naturalness_sources[case] = source
    hashes = {}
    for case in set(selected) | set(naturalness_cases):
        methods = grouped[case]
        if args.baseline not in methods or args.baseline_copy not in methods:
            raise ValueError(f"case {case}: duplicate baseline control is missing")
        files = []
        for method in (args.baseline, args.baseline_copy):
            raw = methods[method].get("audio_file", "")
            if not raw:
                raise ValueError(f"case {case}: {method} has no audio_file")
            path = audio_path(str(raw), results_path)
            if not path.is_file():
                raise FileNotFoundError(f"case {case}: missing {method} audio {path}")
            files.append(path)
        for path in files:
            if path not in hashes:
                hashes[path] = sha256_file(path)
        if hashes[files[0]] != hashes[files[1]]:
            raise ValueError(f"case {case}: duplicate baseline audio differs; packet generation stopped")
    planned = []
    for case in selected:
        methods = grouped[case]
        for method_x, method_y in pairs:
            if method_x not in methods or method_y not in methods:
                raise ValueError(f"incomplete case {case}: missing {method_x} or {method_y}")
            x_raw, y_raw = methods[method_x].get("audio_file", ""), methods[method_y].get("audio_file", "")
            if not x_raw or not y_raw:
                raise ValueError(f"case {case}: both methods need audio_file")
            x_file, y_file = audio_path(str(x_raw), results_path), audio_path(str(y_raw), results_path)
            if not x_file.is_file() or not y_file.is_file():
                raise FileNotFoundError(f"case {case}: missing audio: {x_file} or {y_file}")
            planned.append((case, method_x, method_y, x_file, y_file))
    if args.out_dir.exists():
        raise FileExistsError(f"packet directory already exists: {args.out_dir}")
    rng.shuffle(planned)
    listener = args.out_dir / "listener"
    analyst = args.out_dir / "analyst"
    clips = listener / "clips"
    naturalness_clips = listener / "naturalness_clips"
    clips.mkdir(parents=True)
    naturalness_clips.mkdir()
    analyst.mkdir()
    trials, key, template = [], [], []
    for index, (case, method_x, method_y, x_file, y_file) in enumerate(planned, 1):
        trial_id = f"T{index:05d}"
        a_method, b_method = (method_x, method_y) if rng.getrandbits(1) else (method_y, method_x)
        sources = {method_x: x_file, method_y: y_file}
        clip_a = f"clips/{trial_id}_A{x_file.suffix if a_method == method_x else y_file.suffix}"
        clip_b = f"clips/{trial_id}_B{y_file.suffix if b_method == method_y else x_file.suffix}"
        shutil.copy2(sources[a_method], listener / clip_a)
        shutil.copy2(sources[b_method], listener / clip_b)
        trials.append({"trial_id": trial_id, "clip_a": clip_a, "clip_b": clip_b})
        key.append({"trial_id": trial_id, "sample_id": case[0], "speaker_id": case[1],
                    "seed": case[2], "method_a": a_method, "method_b": b_method})
        template.append({"listener_id": "", "trial_id": trial_id, "choice": ""})
    write_csv(listener / "trials.csv", ("trial_id", "clip_a", "clip_b"), trials)
    write_csv(listener / "ratings_template.csv", ("listener_id", "trial_id", "choice"), template)
    write_csv(analyst / "answer_key.csv", ("trial_id", *CASE_FIELDS, "method_a", "method_b"), key)
    naturalness_rng.shuffle(naturalness_cases)
    naturalness_trials, naturalness_key, naturalness_template = [], [], []
    for index, case in enumerate(naturalness_cases, 1):
        source = naturalness_sources[case]
        trial_id = f"N{index:05d}"
        clip = f"naturalness_clips/{trial_id}{source.suffix}"
        shutil.copy2(source, listener / clip)
        naturalness_trials.append({"trial_id": trial_id, "clip": clip})
        naturalness_key.append({"trial_id": trial_id, "sample_id": case[0],
                                "speaker_id": case[1], "seed": case[2], "method": args.baseline})
        naturalness_template.append({"listener_id": "", "trial_id": trial_id, "rating": ""})
    write_csv(listener / "naturalness_trials.csv", ("trial_id", "clip"), naturalness_trials)
    write_csv(listener / "naturalness_ratings_template.csv",
              ("listener_id", "trial_id", "rating"), naturalness_template)
    write_csv(analyst / "naturalness_answer_key.csv",
              ("trial_id", *CASE_FIELDS, "method"), naturalness_key)
    write_csv(analyst / "selection_manifest.csv", CASE_FIELDS,
              [dict(zip(CASE_FIELDS, case)) for case in selected])
    write_csv(analyst / "naturalness_selection_manifest.csv", CASE_FIELDS,
              [dict(zip(CASE_FIELDS, case)) for case in naturalness_cases])
    (listener / "README.txt").write_text(
        "Task 1: Listen to clip A and clip B for each row of trials.csv. Choose the clip you "
        "prefer for overall speech naturalness, or choose tie. Fill listener_id and choice "
        "(A, B, or tie) in ratings_template.csv.\n\n"
        "Task 2: Listen to each single clip in naturalness_trials.csv. Rate naturalness from "
        "1 (very unnatural) to 5 (very natural) in naturalness_ratings_template.csv. "
        "No reference speech is supplied: this task measures naturalness, not speaker fidelity.\n\n"
        "Do not look for method identities. Save one completed CSV per listener for each task. "
        "Collect at least three independent ratings for every pair and single clip.\n",
        encoding="utf-8",
    )
    (analyst / "README.txt").write_text(
        "Keep answer_key.csv and naturalness_answer_key.csv private until ratings are collected. "
        "Share only the listener directory with raters. selection_manifest.csv records the "
        "preselected A/B cases, and naturalness_selection_manifest.csv records the single-clip "
        "cases. Combine each type of completed rating rows into separate CSVs. "
        "Pass pairwise rows with --ratings/--key and single-clip rows with "
        "--naturalness-ratings/--naturalness-key. Collect at least three independent ratings "
        "per pair and single clip.\n",
        encoding="utf-8",
    )
    print(f"Created {len(trials)} randomized A/B trials from {len(selected)} cases and "
          f"{len(naturalness_trials)} single-clip naturalness trials at {args.out_dir}")
    print(f"Share only {listener} with listeners; keep {analyst} private.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "packet"):
        command = sub.add_parser(name)
        command.add_argument("--results", required=True, type=Path, help="long-form metrics.csv or legacy nested benchmark JSON")
        command.add_argument("--baseline", default="baseline")
        command.add_argument("--fixed", default="fixed")
        command.add_argument("--proposed", default="proposed")
        command.add_argument("--seed", type=int, default=2026, help="bootstrap or packet random seed")
        if name == "analyze":
            command.add_argument("--baseline-copy", default="baseline_copy")
            command.add_argument("--manifest", type=Path, help="held-out prompt manifest for expected-case coverage")
            command.add_argument("--seeds", nargs="+", help="expected synthesis seeds used with --manifest")
            command.add_argument("--ratings", type=Path, help="combined blind ratings CSV")
            command.add_argument("--key", type=Path, help="private packet answer key CSV")
            command.add_argument("--naturalness-ratings", type=Path,
                                 help="combined single-clip naturalness ratings CSV")
            command.add_argument("--naturalness-key", type=Path,
                                 help="private single-clip answer key CSV")
            command.add_argument("--bootstrap", type=int, default=5000)
            command.add_argument("--out", type=Path, help="Markdown report path; stdout if omitted")
        else:
            command.add_argument("--out-dir", required=True, type=Path, help="new packet directory")
            command.add_argument("--baseline-copy", default="baseline_copy")
            command.add_argument("--max-cases", type=int, help="seeded balanced subset size; default all cases")
            command.add_argument("--naturalness-seed", default="1001",
                                 help="one synthesis seed for naturalness clips; default 1001")
            command.add_argument("--naturalness-max-cases", type=int,
                                 help="balanced single-clip subset size; default all complete cases at naturalness seed")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        (analyze if arguments.command == "analyze" else make_packet)(arguments)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
