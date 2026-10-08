"""Prepare a speaker-disjoint LibriSpeech test set for the thesis experiment.

The default selection has twenty speakers and five pairs per speaker. Each pair
uses a different reference utterance of at most four seconds, so inference
does not need to trim reference audio while retaining its full transcript.
Speaker and pair selection use source metadata only; they do not use
synthesis outcomes or benchmark scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIST = ROOT / "data" / "librispeech_pc_test_clean_cross_sentence.lst"
PILOT_MANIFEST = ROOT / "BENCHMARK_100_SAMPLES" / "data" / "manifest_100.json"
ARCHIVE = ROOT / "BENCHMARK_100_SAMPLES" / "data" / "test-clean.tar.gz"
DEFAULT_SPEAKERS = "61,121,260,908,1320,1580,1995,2830,3575,4446,4507,4970,5105,5142,5683,6829,7021,7127,8455,8463"
MAX_REF_DURATION = 4.0
UTT_PATTERN = re.compile(r"^(\d+)-(\d+)-(\d+)$")


def utterance_parts(value: str) -> tuple[str, str, str]:
    match = UTT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid LibriSpeech utterance ID: {value!r}")
    return match.groups()


def score(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def read_source() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(SOURCE_LIST.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"Source line {line_number}: expected six tab-separated fields")
        ref_utt, ref_duration, ref_text, gen_utt, gen_duration, gen_text = fields
        speaker, chapter, _ = utterance_parts(ref_utt)
        gen_speaker, _, _ = utterance_parts(gen_utt)
        if speaker != gen_speaker:
            raise ValueError(f"Source line {line_number}: reference and target speakers differ")
        ref_seconds, gen_seconds = float(ref_duration), float(gen_duration)
        if not all(math.isfinite(v) and v > 0 for v in (ref_seconds, gen_seconds)):
            raise ValueError(f"Source line {line_number}: invalid duration")
        if not ref_text.strip() or not gen_text.strip():
            raise ValueError(f"Source line {line_number}: empty transcript")
        if ref_seconds > MAX_REF_DURATION:
            continue
        relative_audio = f"LibriSpeech/test-clean/{speaker}/{chapter}/{ref_utt}.flac"
        rows.append(
            {
                "speaker_id": speaker,
                "chapter_id": chapter,
                "ref_utt": ref_utt,
                "ref_duration": ref_seconds,
                "ref_text": ref_text,
                "gen_utt": gen_utt,
                "gen_duration": gen_seconds,
                "gen_text": gen_text,
                "rel_flac_path": relative_audio,
            }
        )
    return rows


def choose_rows(rows: list[dict[str, object]], speakers: list[str], per_speaker: int, seed: int) -> list[dict[str, object]]:
    pilot = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    pilot_speakers = {str(row["speaker_id"]) for row in pilot}
    overlap = sorted(set(speakers) & pilot_speakers, key=int)
    if overlap:
        raise ValueError(f"Speakers already used by the pilot manifest: {', '.join(overlap)}")

    chosen: list[dict[str, object]] = []
    for speaker in sorted(speakers, key=int):
        by_reference: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            if row["speaker_id"] == speaker:
                by_reference.setdefault(str(row["ref_utt"]), []).append(row)
        if len(by_reference) < per_speaker:
            raise ValueError(
                f"Speaker {speaker} has {len(by_reference)} unique references; "
                f"{per_speaker} were requested"
            )
        references = sorted(by_reference, key=lambda ref: (score(seed, speaker, "ref", ref), ref))
        for reference in references[:per_speaker]:
            candidates = by_reference[reference]
            picked = min(
                candidates,
                key=lambda row: (score(seed, speaker, "target", row["gen_utt"]), str(row["gen_utt"])),
            )
            chosen.append(picked)

    targets = [str(row["gen_utt"]) for row in chosen]
    if len(targets) != len(set(targets)):
        raise ValueError("Selection contains a repeated target utterance")
    return chosen


def relative_or_absolute(path: Path, from_directory: Path) -> str:
    try:
        return os.path.relpath(path, from_directory).replace("\\", "/")
    except ValueError:  # Different Windows drives.
        return path.as_posix()


def build_manifest(rows: list[dict[str, object]], out: Path, data_dir: Path) -> list[dict[str, object]]:
    manifest = []
    for sample_id, row in enumerate(rows, 1):
        item = {"sample_id": sample_id, **row}
        audio_path = data_dir / Path(str(row["rel_flac_path"]))
        item["ref_audio_path"] = relative_or_absolute(audio_path, out.parent)
        manifest.append(item)
    return manifest


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_references(archive: Path, manifest: list[dict[str, object]], data_dir: Path) -> tuple[int, int]:
    expected = {str(row["rel_flac_path"]): data_dir / Path(str(row["rel_flac_path"])) for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("Selected rows must have distinct reference audio files")

    found: set[str] = set()
    written = reused = 0
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            if member.name not in expected:
                continue
            if member.name in found:
                raise ValueError(f"Duplicate archive member: {member.name}")
            if not member.isfile() or member.size <= 0:
                raise ValueError(f"Archive member is not a regular nonempty file: {member.name}")
            found.add(member.name)
            destination = expected[member.name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            audio = source.extractfile(member)
            if audio is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            with audio:
                if destination.exists():
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                        digest.update(chunk)
                    if destination.stat().st_size != member.size or file_digest(destination) != digest.hexdigest():
                        raise ValueError(f"Existing reference differs from archive: {destination}")
                    reused += 1
                    continue

                temporary_name: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as temporary:
                        temporary_name = temporary.name
                        shutil.copyfileobj(audio, temporary)
                    if Path(temporary_name).stat().st_size != member.size:
                        raise ValueError(f"Incomplete extraction: {member.name}")
                    os.replace(temporary_name, destination)
                    written += 1
                finally:
                    if temporary_name is not None:
                        Path(temporary_name).unlink(missing_ok=True)

    missing = set(expected) - found
    if missing:
        raise ValueError(f"Archive lacks selected reference audio: {', '.join(sorted(missing))}")
    return written, reused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "THESIS_EXPERIMENT/data/holdout_manifest.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "THESIS_EXPERIMENT/data/references")
    parser.add_argument("--speakers", default=DEFAULT_SPEAKERS, help="Comma-separated, unused LibriSpeech speaker IDs")
    parser.add_argument("--per-speaker", type=int, default=5, help="Pairs with distinct references per speaker")
    parser.add_argument("--seed", type=int, default=20260925, help="Deterministic pair-selection seed")
    parser.add_argument("--dry-run", action="store_true", help="Report selection without writing files")
    parser.add_argument("--replace-manifest", action="store_true", help="Replace an existing manifest when the selection changes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_speaker <= 0:
        raise ValueError("--per-speaker must be positive")
    speakers = [speaker.strip() for speaker in args.speakers.split(",")]
    if not speakers or any(not speaker.isdecimal() for speaker in speakers) or len(speakers) != len(set(speakers)):
        raise ValueError("--speakers must be a comma-separated list of distinct numeric IDs")
    out, data_dir = args.out.expanduser().resolve(), args.data_dir.expanduser().resolve()
    selected = choose_rows(read_source(), speakers, args.per_speaker, args.seed)
    manifest = build_manifest(selected, out, data_dir)
    summary = f"Selected {len(manifest)} pairs from {len(speakers)} new speakers, with unique reference audio per pair"
    print(summary)
    if args.dry_run:
        return 0

    if out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != manifest:
            if not args.replace_manifest:
                raise ValueError(f"Existing manifest differs from this selection: {out}; use --replace-manifest to replace it")
            manifest_exists = False
        else:
            manifest_exists = True
    else:
        manifest_exists = False

    written, reused = extract_references(ARCHIVE, manifest, data_dir)
    if not manifest_exists:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, suffix=".tmp", delete=False) as temporary:
                    temporary_name = temporary.name
                    json.dump(manifest, temporary, indent=2, ensure_ascii=False)
                    temporary.write("\n")
                os.replace(temporary_name, out)
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
        else:
            with out.open("x", encoding="utf-8") as destination:
                json.dump(manifest, destination, indent=2, ensure_ascii=False)
                destination.write("\n")
    print(f"Reference audio: {written} extracted, {reused} already present and verified")
    print(f"Manifest: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
