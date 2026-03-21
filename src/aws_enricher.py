from __future__ import annotations

import json
import os
import time
import uuid
import urllib.request
from typing import Optional

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def _require_boto3() -> None:
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required for AWS features. Install with: pip install boto3"
        )


def upload_to_s3(audio_path: str, bucket: str, key: str, region: str) -> str:
    _require_boto3()
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(audio_path, bucket, key)
    return f"s3://{bucket}/{key}"


def transcribe_audio(
    audio_path: str,
    s3_bucket: str,
    region: str = "us-east-1",
    job_prefix: str = "music-analyzer",
    poll_interval: int = 5,
    timeout: int = 300,
) -> str:
    """Upload audio to S3, run AWS Transcribe, return transcript text."""
    _require_boto3()

    ext = os.path.splitext(audio_path)[1].lstrip(".").lower()
    media_format_map = {
        "mp3": "mp3",
        "wav": "wav",
        "m4a": "mp4",
        "mp4": "mp4",
        "flac": "flac",
    }
    media_format = media_format_map.get(ext, "wav")

    job_name = f"{job_prefix}-{uuid.uuid4().hex[:8]}"
    s3_key = f"transcribe-input/{job_name}/{os.path.basename(audio_path)}"
    s3_uri = upload_to_s3(audio_path, s3_bucket, s3_key, region)

    transcribe = boto3.client("transcribe", region_name=region)
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat=media_format,
        LanguageCode="en-US",
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            transcript_uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            with urllib.request.urlopen(transcript_uri) as r:
                data = json.loads(r.read().decode())
            return data["results"]["transcripts"][0]["transcript"]
        elif status == "FAILED":
            reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
            raise RuntimeError(f"Transcription job failed: {reason}")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Transcription job {job_name} did not complete within {timeout}s"
    )


def comprehend_analyze(text: str, region: str = "us-east-1") -> dict:
    """Run AWS Comprehend entity detection, key-phrase extraction, and sentiment analysis."""
    _require_boto3()

    if not text.strip():
        return {
            "entities": [],
            "key_phrases": [],
            "sentiment": "NEUTRAL",
            "sentiment_scores": {},
        }

    comprehend = boto3.client("comprehend", region_name=region)
    # Comprehend has a 5000-byte input limit
    text_truncated = text[:4900]

    entities_resp = comprehend.detect_entities(Text=text_truncated, LanguageCode="en")
    phrases_resp = comprehend.detect_key_phrases(Text=text_truncated, LanguageCode="en")
    sentiment_resp = comprehend.detect_sentiment(Text=text_truncated, LanguageCode="en")

    return {
        "entities": entities_resp.get("Entities", []),
        "key_phrases": [kp["Text"] for kp in phrases_resp.get("KeyPhrases", [])],
        "sentiment": sentiment_resp.get("Sentiment", "NEUTRAL"),
        "sentiment_scores": sentiment_resp.get("SentimentScore", {}),
    }


_GENRE_KEYWORDS = {
    "pop", "rock", "jazz", "blues", "hip hop", "rap", "r&b", "soul",
    "country", "folk", "classical", "electronic", "dance", "edm",
    "metal", "punk", "reggae", "latin", "indie", "alternative",
}


def extract_metadata_signals(comprehend_result: dict) -> dict:
    """
    Extract song title, artist, and genre signals from Comprehend output.

    Returns dict with keys:
      title, artist, genre, tags,
      transcript_entity_count, transcript_phrase_count
    """
    entities = comprehend_result.get("entities", [])
    key_phrases = comprehend_result.get("key_phrases", [])
    sentiment = comprehend_result.get("sentiment", "NEUTRAL")
    sentiment_scores = comprehend_result.get("sentiment_scores", {})

    title_candidates: list[str] = []
    artist_candidates: list[str] = []
    genre_candidates: list[str] = []

    for ent in entities:
        etype = ent.get("Type", "")
        etext = ent.get("Text", "")
        if etype == "PERSON":
            artist_candidates.append(etext)
        elif etype == "TITLE":
            title_candidates.append(etext)
        elif etype == "OTHER":
            if any(g in etext.lower() for g in _GENRE_KEYWORDS):
                genre_candidates.append(etext)

    for phrase in key_phrases:
        if any(g in phrase.lower() for g in _GENRE_KEYWORDS):
            genre_candidates.append(phrase)

    # Build descriptive category tags from sentiment + genre signals
    tags: list[str] = []
    if sentiment == "POSITIVE":
        tags.append("upbeat")
    elif sentiment == "NEGATIVE":
        tags.append("somber")
    elif sentiment == "MIXED":
        tags.append("complex-mood")

    pos_score = sentiment_scores.get("Positive", 0.0)
    neg_score = sentiment_scores.get("Negative", 0.0)
    if pos_score > 0.7:
        tags.append("high-energy")
    if neg_score > 0.5:
        tags.append("emotional")

    for g in genre_candidates[:2]:
        tags.append(g.lower())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tags: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)

    return {
        "title": title_candidates[0] if title_candidates else None,
        "artist": artist_candidates[0] if artist_candidates else None,
        "genre": genre_candidates[0] if genre_candidates else None,
        "tags": unique_tags,
        "sentiment": sentiment,
        "transcript_entity_count": len(entities),
        "transcript_phrase_count": len(key_phrases),
    }


def enrich_track(
    audio_path: str,
    s3_bucket: str,
    region: str = "us-east-1",
) -> dict:
    """Full AWS enrichment pipeline for a single track.

    Returns metadata signals dict (title, artist, genre, tags, sentiment, …)
    plus the raw transcript text.
    """
    transcript = transcribe_audio(audio_path, s3_bucket, region=region)
    comprehend_result = comprehend_analyze(transcript, region=region)
    signals = extract_metadata_signals(comprehend_result)
    signals["transcript"] = transcript
    return signals
