# Music-Accompaniment---AI-powered

🎵 Music Analyzer — AI-Powered Audio Analysis (Python, PyTorch, Librosa, AWS Transcribe, AWS Comprehend)

An end-to-end Python CLI pipeline that ingests raw audio files (wav/mp3/m4a), extracts harmonic chroma and mel-spectrogram features via Librosa, runs a custom PyTorch CNN for 24-class musical key classification, and enriches each track's report using AWS Transcribe + Comprehend NLP. All results are aggregated into a structured CSV batch report per run.

## ✨ Features

**🎼 Musical Key Detection**
- Template-based (chroma + Krumhansl–Schmuckler key profiles)
- Neural network–based (custom CNN trained on mel-spectrograms, 24-class: 12 major + 12 minor)

**🧠 AI Neural Key Classifier**
- Custom-trained CNN (PyTorch)
- 24-class classification (12 major + 12 minor keys)
- Input: log-mel spectrogram (128 mel bins, 12-second clips)

**🎹 Accompaniment Suggestions**
- Recommended scales & modes (Ionian, Mixolydian, Dorian, Harmonic Minor, etc.)
- Target chord tones to emphasize
- Chroma-driven mode hints (e.g., strong ♭7 → Mixolydian)

**☁️ AWS Transcribe + Comprehend Enrichment**
- Converts audio content to text via AWS Transcribe
- Passes transcript to AWS Comprehend for NLP entity detection and key-phrase extraction
- Automatically surfaces song title, artist, and genre signals from unstructured audio metadata
- Sentiment scores and entity labels used to generate descriptive category tags (e.g., `upbeat`, `high-energy`, `jazz`)

**📊 CSV Batch Report**
- Aggregates all outputs into a structured CSV per run
- Flags which tracks lacked sufficient metadata coverage (title / artist / genre signals)
- Enables data-driven analysis of the full audio catalog

**⏱ Tempo (BPM) Estimation**
- Optional beat tracking for rhythmic context

## 🧠 How It Works

1. **Audio Processing**
   - Load audio (wav/mp3/m4a/flac)
   - Normalize & trim
   - Extract features:
     - Template method → Harmonic chroma (CQT-based)
     - Neural method → Log mel-spectrogram

2. **Key Detection**
   - Template detector: cosine similarity against major/minor key profiles
   - Neural detector: CNN predicts key probabilities → top-1 key + confidence margin

3. **Musical Intelligence**
   - Select appropriate scales (Ionian, Pentatonic, Dorian, etc.)
   - Identify chord tones (1–3–5 or 1–♭3–5)
   - Output musician-friendly guidance

4. **AWS Enrichment (optional)**
   - Upload audio to S3 → AWS Transcribe job → transcript text
   - AWS Comprehend: `detect_entities` + `detect_key_phrases` + `detect_sentiment`
   - Extract PERSON (artist), TITLE, genre keywords from entities and key phrases
   - Combine sentiment scores with local ML predictions → category tags

5. **Batch CSV Report**
   - Every run writes `out/batch_report.csv`
   - Columns: file, key, mode, confidence, bpm, scales, title/artist/genre signals, sentiment, tags, metadata_coverage

## 🏗 Project Structure

```
Music-Accompaniment---AI-powered/
│
├── src/
│   ├── analyze.py        # Main CLI entry point (single + batch mode)
│   ├── neural_key.py     # Neural model inference (mel → key)
│   ├── aws_enricher.py   # AWS Transcribe + Comprehend integration
│   └── __init__.py
│
├── ml/
│   ├── scripts/
│   │   ├── dataset.py    # Audio → mel-spectrogram dataset
│   │   ├── model.py      # KeyCNN architecture (24-class)
│   │   ├── label_map.py  # key ↔ class-id mapping
│   │   ├── train.py      # Training loop (AdamW, CrossEntropy)
│   │   └── test_dataset.py
│   ├── data/             # (ignored) training audio
│   ├── models/           # (ignored) checkpoints
│   └── runs/
│
├── input/                # Example audio files
├── out/                  # Per-track reports + batch_report.csv
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
git clone https://github.com/Tinle0301/Music-Analyzer---AI-powered.git
cd Music-Analyzer---AI-powered

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Optional: better audio decoding
brew install ffmpeg
```

For AWS features, configure credentials via `aws configure` or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

## ▶️ Usage

**Single file — template key detection**
```bash
python3 -m src.analyze input/song.m4a --auto-bpm
```

**Single file — neural key detection**
```bash
python3 -m src.analyze input/song.m4a \
  --auto-bpm \
  --detector neural \
  --ckpt ml/models/key_cnn_best.pt
```

**Batch mode (directory or multiple files)**
```bash
python3 -m src.analyze input/ --auto-bpm --csv-report out/run_report.csv
```

**With AWS Transcribe + Comprehend enrichment**
```bash
python3 -m src.analyze input/song.wav \
  --aws \
  --s3-bucket my-bucket \
  --aws-region us-east-1
```

**Output example**
```
Detected key: E major  (score=0.847, confidence=0.123)
Relative minor: C# minor
Estimated BPM (rough): 112.0

Recommended scales / modes (for accompaniment):
  1. Major (Ionian) — Default safe choice for major-key accompaniment.
  2. Major Pentatonic — Easy melodic accompaniment; avoids harsh tensions.

Target notes (emphasize these): E, G#, B

AWS Enrichment:
  Artist signal: Taylor Swift
  Genre signal:  pop
  Sentiment:     POSITIVE
  Category tags: upbeat, high-energy, pop
  Entities detected: 4, Key phrases: 7
```

All reports saved to `out/` as `<song>_report.txt` plus `out/batch_report.csv`.

## 🧪 Training the Neural Model

Prepare a manifest (`ml/data/manifest.tsv`):
```
ml/data/audio/track001.m4a	E	major
ml/data/audio/track002.m4a	C#	minor
```

Train:
```bash
python3 -m ml.scripts.train --epochs 20 --batch_size 8
```

Checkpoints saved to `ml/models/key_cnn_best.pt`.

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Audio features | Librosa (chroma CQT, mel-spectrogram, beat tracking) |
| ML / Key CNN | PyTorch (custom CNN, AdamW, CrossEntropy) |
| NLP / Metadata | AWS Transcribe, AWS Comprehend |
| Data pipeline | NumPy, SciPy, Scikit-learn |
| Reporting | Python csv module, argparse CLI |

## 📜 License

MIT License
