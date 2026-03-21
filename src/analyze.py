#!/usr/bin/env python3
"""
analyze.py
Analyze one or more audio files and output per-track reports plus a CSV batch summary.

Per-track report includes:
- Estimated musical key (major/minor) + confidence
- Recommended scales/modes for accompaniment
- Target notes (chord tones) to emphasize
- Optional BPM estimate
- Optional AWS-enriched metadata: title/artist/genre signals, sentiment, category tags

CSV batch report aggregates all tracks and flags which lacked sufficient metadata.

Usage:
  # Single file
  python src/analyze.py input/song.wav

  # Multiple files / batch
  python src/analyze.py input/song1.wav input/song2.mp3 --auto-bpm

  # With AWS Transcribe + Comprehend enrichment
  python src/analyze.py input/song.wav --aws --s3-bucket my-bucket --aws-region us-east-1

  # Save batch CSV to custom path
  python src/analyze.py input/ --csv-report out/run_report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
import torch

from src.neural_key import load_model as load_neural_model, predict_key as predict_neural_key


# ----------------------------
# Music theory helpers
# ----------------------------
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyResult:
    tonic: str          # e.g., "E"
    mode: str           # "major" or "minor"
    score: float        # similarity score
    confidence: float   # top1 - top2 (simple margin)


@dataclass
class Recommendation:
    name: str
    reason: str


# ----------------------------
# Audio + feature extraction
# ----------------------------
def load_audio(path: str, sr: int = 22050, duration: float | None = None) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Audio loaded but is empty.")
    y = librosa.util.normalize(y)
    return y, sr


def harmonic_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    y_harm, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # 12-dim
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    return chroma_mean


# ----------------------------
# Key detection (template matching)
# ----------------------------
def rotate(v: np.ndarray, n: int) -> np.ndarray:
    return np.roll(v, n)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def detect_key(chroma_mean: np.ndarray) -> KeyResult:
    maj = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    minr = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)

    scores = []
    for k in range(12):
        maj_k = rotate(maj, k)
        min_k = rotate(minr, k)
        scores.append(("major", k, cosine_sim(chroma_mean, maj_k)))
        scores.append(("minor", k, cosine_sim(chroma_mean, min_k)))

    scores_sorted = sorted(scores, key=lambda x: x[2], reverse=True)
    top_mode, top_k, top_score = scores_sorted[0]
    _, _, second_score = scores_sorted[1]

    tonic = NOTE_NAMES_SHARP[top_k]
    confidence = float(top_score - second_score)
    return KeyResult(tonic=tonic, mode=top_mode, score=float(top_score), confidence=confidence)


# ----------------------------
# Scale recommendations for accompaniment
# ----------------------------
def note_index(name: str) -> int:
    return NOTE_NAMES_SHARP.index(name)


def pitchclass_name(pc: int) -> str:
    return NOTE_NAMES_SHARP[pc % 12]


def build_scale(tonic_pc: int, intervals: List[int]) -> List[str]:
    return [pitchclass_name(tonic_pc + i) for i in intervals]


def recommend_scales(key: KeyResult, chroma_mean: np.ndarray) -> List[Recommendation]:
    tonic_pc = note_index(key.tonic)
    recs: List[Recommendation] = []

    if key.mode == "major":
        recs.append(Recommendation("Major (Ionian)", "Default safe choice for major-key accompaniment."))
        recs.append(Recommendation("Major Pentatonic", "Easy melodic accompaniment; avoids harsh tensions."))

        b7_pc = (tonic_pc + 10) % 12
        if chroma_mean[b7_pc] > 0.28:
            recs.append(Recommendation("Mixolydian", "Detected strong ♭7 color; common in rock/pop grooves."))

        sharp4_pc = (tonic_pc + 6) % 12
        if chroma_mean[sharp4_pc] > 0.22:
            recs.append(Recommendation("Lydian", "Detected strong ♯4 color; gives bright cinematic sound."))
    else:
        recs.append(Recommendation("Natural Minor (Aeolian)", "Default safe choice for minor-key accompaniment."))
        recs.append(Recommendation("Minor Pentatonic", "Most common for minor accompaniment and riffs."))
        recs.append(Recommendation("Blues Scale", "Works for expressive fills; especially pop/R&B/rock."))

        nat6_pc = (tonic_pc + 9) % 12
        if chroma_mean[nat6_pc] > 0.25:
            recs.append(Recommendation("Dorian", "Detected strong natural 6; funk/jazz minor sound."))

        maj7_pc = (tonic_pc + 11) % 12
        if chroma_mean[maj7_pc] > 0.22:
            recs.append(Recommendation("Harmonic Minor", "Detected leading-tone pull; dramatic V→i sound."))

    return recs


def target_notes(key: KeyResult) -> List[str]:
    tonic_pc = note_index(key.tonic)
    if key.mode == "major":
        pcs = [tonic_pc, tonic_pc + 4, tonic_pc + 7]
    else:
        pcs = [tonic_pc, tonic_pc + 3, tonic_pc + 7]
    return [pitchclass_name(pc) for pc in pcs]


# ----------------------------
# Optional tempo estimate
# ----------------------------
def estimate_bpm(y: np.ndarray, sr: int) -> float:
    tempo = librosa.beat.tempo(y=y, sr=sr)
    return float(tempo[0]) if len(tempo) else 0.0


# ----------------------------
# Reporting
# ----------------------------
def format_report(
    path: str,
    key: KeyResult,
    bpm: float | None,
    recs: List[Recommendation],
    aws_signals: Optional[Dict[str, Any]] = None,
) -> str:
    base = os.path.basename(path)

    if key.mode == "major":
        rel_minor = pitchclass_name(note_index(key.tonic) + 9)
        rel = f"Relative minor: {rel_minor} minor"
    else:
        rel_major = pitchclass_name(note_index(key.tonic) + 3)
        rel = f"Relative major: {rel_major} major"

    lines = []
    lines.append(f"Song: {base}")
    lines.append("")
    lines.append(f"Detected key: {key.tonic} {key.mode}  (score={key.score:.3f}, confidence={key.confidence:.3f})")
    lines.append(rel)
    if bpm is not None:
        lines.append(f"Estimated BPM (rough): {bpm:.1f}")
    lines.append("")
    lines.append("Recommended scales / modes (for accompaniment):")
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r.name} — {r.reason}")
    lines.append("")
    lines.append(f"Target notes (emphasize these): {', '.join(target_notes(key))}")

    if aws_signals:
        lines.append("")
        lines.append("AWS Enrichment:")
        if aws_signals.get("title"):
            lines.append(f"  Title signal:  {aws_signals['title']}")
        if aws_signals.get("artist"):
            lines.append(f"  Artist signal: {aws_signals['artist']}")
        if aws_signals.get("genre"):
            lines.append(f"  Genre signal:  {aws_signals['genre']}")
        sentiment = aws_signals.get("sentiment", "")
        if sentiment:
            lines.append(f"  Sentiment:     {sentiment}")
        tags = aws_signals.get("tags", [])
        if tags:
            lines.append(f"  Category tags: {', '.join(tags)}")
        lines.append(
            f"  Entities detected: {aws_signals.get('transcript_entity_count', 0)}, "
            f"Key phrases: {aws_signals.get('transcript_phrase_count', 0)}"
        )

    lines.append("")
    lines.append("Tip: Start with the top scale. If it feels 'off', try the next suggested mode.")
    return "\n".join(lines)


def write_csv_report(records: List[Dict[str, Any]], output_path: str) -> None:
    """Write a structured CSV batch report aggregating all track results."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "file",
        "key",
        "mode",
        "score",
        "confidence",
        "bpm",
        "scale_1",
        "scale_2",
        "target_notes",
        "title_signal",
        "artist_signal",
        "genre_signal",
        "sentiment",
        "category_tags",
        "transcript_entity_count",
        "transcript_phrase_count",
        "metadata_coverage",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV batch report -> {output_path}")


def _build_record(
    path: str,
    key: KeyResult,
    bpm: float | None,
    recs: List[Recommendation],
    aws_signals: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """Build a flat dict suitable for CSV output."""
    aws = aws_signals or {}
    title = aws.get("title") or ""
    artist = aws.get("artist") or ""
    genre = aws.get("genre") or ""

    # metadata_coverage: fraction of [title, artist, genre] that were found
    aws_fields = [title, artist, genre]
    if aws:
        filled = sum(1 for v in aws_fields if v)
        coverage = f"{filled}/3"
    else:
        coverage = "no_aws"

    return {
        "file": os.path.basename(path),
        "key": key.tonic if not error else "",
        "mode": key.mode if not error else "",
        "score": f"{key.score:.3f}" if not error else "",
        "confidence": f"{key.confidence:.3f}" if not error else "",
        "bpm": f"{bpm:.1f}" if bpm is not None else "",
        "scale_1": recs[0].name if recs else "",
        "scale_2": recs[1].name if len(recs) > 1 else "",
        "target_notes": ", ".join(target_notes(key)) if not error else "",
        "title_signal": title,
        "artist_signal": artist,
        "genre_signal": genre,
        "sentiment": aws.get("sentiment", ""),
        "category_tags": ", ".join(aws.get("tags", [])),
        "transcript_entity_count": aws.get("transcript_entity_count", ""),
        "transcript_phrase_count": aws.get("transcript_phrase_count", ""),
        "metadata_coverage": coverage,
        "error": error,
    }


def _collect_audio_files(inputs: List[str]) -> List[str]:
    """Expand directories; return sorted list of audio file paths."""
    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".ogg"}
    files: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fname in sorted(os.listdir(inp)):
                if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(inp, fname))
        else:
            files.append(inp)
    return files


def _pick_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(
        description="Analyze audio -> key + scale recommendations. Optionally enrich via AWS."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to audio file(s) (wav/mp3/m4a/flac) or a directory.",
    )
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--time", type=float, default=None, help="Max seconds of audio to analyze")
    ap.add_argument("--auto-bpm", action="store_true", help="Estimate BPM")
    ap.add_argument(
        "--detector",
        choices=["template", "neural"],
        default="template",
        help="Key detector backend",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="ml/models/key_cnn_best.pt",
        help="Neural model checkpoint (used with --detector neural)",
    )

    # AWS enrichment
    ap.add_argument(
        "--aws",
        action="store_true",
        help="Enable AWS Transcribe + Comprehend enrichment",
    )
    ap.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket name for AWS Transcribe uploads (required with --aws)",
    )
    ap.add_argument(
        "--aws-region",
        type=str,
        default="us-east-1",
        help="AWS region for Transcribe + Comprehend",
    )

    # Batch CSV output
    ap.add_argument(
        "--csv-report",
        type=str,
        default="out/batch_report.csv",
        help="Path to write the structured CSV batch report",
    )

    args = ap.parse_args()

    if args.aws and not args.s3_bucket:
        ap.error("--s3-bucket is required when using --aws")

    audio_files = _collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Load neural model once if needed
    neural_model = None
    neural_device = None
    if args.detector == "neural":
        if not os.path.exists(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
            sys.exit(1)
        neural_device = _pick_device()
        neural_model = load_neural_model(args.ckpt, device=neural_device)

    # Lazy import of aws_enricher only when --aws is requested
    enrich_track = None
    if args.aws:
        from src.aws_enricher import enrich_track as _enrich_track
        enrich_track = _enrich_track

    os.makedirs("out", exist_ok=True)
    csv_records: List[Dict[str, Any]] = []

    for audio_path in audio_files:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path}")
        print('='*60)

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"File not found: {audio_path}")

            y, sr = load_audio(audio_path, sr=args.sr, duration=args.time)

            # Key detection
            if args.detector == "template":
                chroma_mean = harmonic_chroma(y, sr)
                key = detect_key(chroma_mean)
            else:
                chroma_mean = None
                tonic, mode, prob, margin = predict_neural_key(
                    neural_model, audio_path, device=neural_device
                )
                key = KeyResult(tonic=tonic, mode=mode, score=prob, confidence=margin)

            # Tempo
            bpm = estimate_bpm(y, sr) if args.auto_bpm else None

            # Scale recommendations
            recs = recommend_scales(
                key, chroma_mean if chroma_mean is not None else np.zeros(12)
            )

            # AWS enrichment (fails gracefully — key/tempo/scale still reported)
            aws_signals: Optional[Dict[str, Any]] = None
            if enrich_track is not None:
                try:
                    print("  Running AWS Transcribe + Comprehend...")
                    aws_signals = enrich_track(
                        audio_path,
                        s3_bucket=args.s3_bucket,
                        region=args.aws_region,
                    )
                except Exception as aws_exc:
                    print(f"  AWS enrichment failed: {aws_exc}", file=sys.stderr)

            # Per-track text report
            report = format_report(audio_path, key, bpm, recs, aws_signals)
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join("out", f"{base}_report.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(report)
            print(f"\nSaved report -> {out_path}")

            csv_records.append(_build_record(audio_path, key, bpm, recs, aws_signals))

        except Exception as exc:
            error_msg = str(exc)
            print(f"  ERROR: {error_msg}", file=sys.stderr)
            # Append a failed row so CSV covers all files
            empty_key = KeyResult(tonic="", mode="", score=0.0, confidence=0.0)
            csv_records.append(
                _build_record(audio_path, empty_key, None, [], error=error_msg)
            )

    # Always write CSV report (shows metadata coverage gaps)
    write_csv_report(csv_records, args.csv_report)

    # Summary
    total = len(csv_records)
    errors = sum(1 for r in csv_records if r.get("error"))
    aws_enriched = sum(1 for r in csv_records if r.get("sentiment"))
    print(f"\nBatch summary: {total} track(s), {errors} error(s), {aws_enriched} AWS-enriched.")


if __name__ == "__main__":
    main()
