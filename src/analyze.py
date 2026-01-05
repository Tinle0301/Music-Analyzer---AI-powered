#!/usr/bin/env python3
"""
analyze.py
Analyze an audio file and output:
- Estimated musical key (major/minor) + confidence
- Recommended scales/modes for accompaniment
- Target notes (chord tones) to emphasize
- Optional BPM estimate (rough)

Usage:
  python src/analyze.py input/song.wav
  python src/analyze.py input/song.wav --time 60 --auto-bpm
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa


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
    # Separate harmonic part to reduce drum influence
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
    # Normalize key profiles
    maj = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    minr = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)

    scores = []
    # For each tonic (0..11), compute similarity for major and minor
    for k in range(12):
        maj_k = rotate(maj, k)
        min_k = rotate(minr, k)
        scores.append(("major", k, cosine_sim(chroma_mean, maj_k)))
        scores.append(("minor", k, cosine_sim(chroma_mean, min_k)))

    # Sort descending by score
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
    """
    Recommend scales based on detected key + chroma hints.
    Simple, practical accompaniment rules.
    """
    tonic_pc = note_index(key.tonic)

    recs: List[Recommendation] = []

    # Basic go-to scales
    if key.mode == "major":
        recs.append(Recommendation("Major (Ionian)", "Default safe choice for major-key accompaniment."))
        recs.append(Recommendation("Major Pentatonic", "Easy melodic accompaniment; avoids harsh tensions."))

        # Hints for modes:
        # strong b7 in major => Mixolydian vibe
        b7_pc = (tonic_pc + 10) % 12
        if chroma_mean[b7_pc] > 0.28:
            recs.append(Recommendation("Mixolydian", "Detected strong ♭7 color; common in rock/pop grooves."))

        # strong #4 in major => Lydian vibe
        sharp4_pc = (tonic_pc + 6) % 12
        if chroma_mean[sharp4_pc] > 0.22:
            recs.append(Recommendation("Lydian", "Detected strong ♯4 color; gives bright cinematic sound."))

    else:
        recs.append(Recommendation("Natural Minor (Aeolian)", "Default safe choice for minor-key accompaniment."))
        recs.append(Recommendation("Minor Pentatonic", "Most common for minor accompaniment and riffs."))
        recs.append(Recommendation("Blues Scale", "Works for expressive fills; especially pop/R&B/rock."))

        # Dorian hint: natural 6 is strong in minor
        nat6_pc = (tonic_pc + 9) % 12  # 6 in minor is +9 semitones from tonic
        if chroma_mean[nat6_pc] > 0.25:
            recs.append(Recommendation("Dorian", "Detected strong natural 6; funk/jazz minor sound."))

        # Harmonic minor hint: leading tone (major 7) appears
        maj7_pc = (tonic_pc + 11) % 12
        if chroma_mean[maj7_pc] > 0.22:
            recs.append(Recommendation("Harmonic Minor", "Detected leading-tone pull; dramatic V→i sound."))

    return recs


def target_notes(key: KeyResult) -> List[str]:
    tonic_pc = note_index(key.tonic)
    if key.mode == "major":
        # 1, 3, 5
        pcs = [tonic_pc, tonic_pc + 4, tonic_pc + 7]
    else:
        # 1, b3, 5
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
def format_report(path: str, key: KeyResult, bpm: float | None, recs: List[Recommendation]) -> str:
    base = os.path.basename(path)
    rel = ""
    if key.mode == "major":
        # relative minor = down 3 semitones
        rel_minor = pitchclass_name(note_index(key.tonic) + 9)
        rel = f"Relative minor: {rel_minor} minor"
    else:
        # relative major = up 3 semitones
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
    lines.append("")
    lines.append("Tip: Start with the top scale. If it feels 'off', try the next suggested mode.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze audio -> key + scale recommendations for accompaniment.")
    ap.add_argument("input", help="Path to audio file (wav/mp3/aiff...)")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate for analysis")
    ap.add_argument("--time", type=float, default=None, help="Analyze only first N seconds (faster)")
    ap.add_argument("--auto-bpm", action="store_true", help="Estimate BPM (rough)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    y, sr = load_audio(args.input, sr=args.sr, duration=args.time)
    chroma_mean = harmonic_chroma(y, sr)
    key = detect_key(chroma_mean)

    bpm = estimate_bpm(y, sr) if args.auto_bpm else None
    recs = recommend_scales(key, chroma_mean)

    report = format_report(args.input, key, bpm, recs)

    os.makedirs("out", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join("out", f"{base}_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nSaved report -> {out_path}")


if __name__ == "__main__":
    main()
