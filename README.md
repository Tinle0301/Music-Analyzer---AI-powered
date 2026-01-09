# Music-Accompaniment---AI-powered

🎵 Music Analyzer — AI-Powered Accompaniment Assistant
An AI-powered music analysis tool that converts audio into musical understanding — detecting key, tempo, and suggesting scales and chord tones for piano or instrumental accompaniment.
This project combines signal processing, music theory, and machine learning (CNNs) to help musicians quickly find the right harmonic framework to play along with a song.

✨ Features

🎼 Musical Key Detection
Template-based (chroma + music theory profiles)
Neural network–based (CNN trained on mel-spectrograms)

🧠 AI Neural Key Classifier
Custom-trained CNN (PyTorch)
24-class classification (12 major + 12 minor)

🎹 Accompaniment Suggestions
Recommended scales & modes
Target chord tones to emphasize

⏱ Tempo (BPM) Estimation
Optional beat tracking for rhythmic context

🧩 Modular CLI Design
Switch between template and neural detectors
Clean separation of audio, ML, and theory logic

🧠 How It Works (High Level)

1. Audio Processing
   Load audio (wav/mp3/m4a)
   Normalize & trim
   Extract features:
   Template method → Harmonic chroma
   Neural method → Log mel-spectrogram
2. Key Detection
   Template detector
   Cosine similarity against major/minor key profiles
   Neural detector
   CNN trained on labeled audio clips
   Predicts key probabilities
3. Musical Intelligence
   Select appropriate scales (Ionian, Pentatonic, Dorian, etc.)
   Identify chord tones (1–3–5 or 1–♭3–5)

Output musician-friendly guidance
🏗 Project Structure
Music-Analyzer---AI-powered/
│
├── src/
│ ├── analyze.py # Main CLI entry
│ ├── neural_key.py # Neural inference logic
│ └── **init**.py
│
├── ml/
│ ├── scripts/
│ │ ├── dataset.py # Audio → mel dataset
│ │ ├── model.py # CNN architecture
│ │ ├── train.py # Training loop
│ │ └── test_dataset.py
│ ├── data/ # (ignored) training audio
│ ├── models/ # (ignored) checkpoints
│ └── runs/
│
├── input/ # Example audio files
├── out/ # Analysis reports
├── requirements.txt
└── README.md

🚀 Installation (macOS + VSCode)
git clone https://github.com/Tinle0301/Music-Analyzer---AI-powered.git
cd Music-Analyzer---AI-powered

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
Optional (recommended for audio decoding):
brew install ffmpeg

▶️ Usage
Template-based key detection
python3 -m src.analyze input/song.m4a --auto-bpm
Neural (AI) key detection
python3 -m src.analyze input/song.m4a \
 --auto-bpm \
 --detector neural \
 --ckpt ml/models/key_cnn_best.pt
Output example
Detected key: E major
Relative minor: C# minor
Estimated BPM: 112

Recommended scales:

1. Major (Ionian)
2. Major Pentatonic

Target notes: E, G#, B
A report is also saved to:
out/song_report.txt

🧪 Training the Neural Model
Prepare a manifest:
ml/data/manifest.tsv

Example:
ml/data/audio/track001.m4a E major
ml/data/audio/track002.m4a C# minor

Train:
python3 -m ml.scripts.train --epochs 20 --batch_size 8

Checkpoints are saved to:
ml/models/key_cnn_best.pt

🛠 Tech Stack
Python 3.12
PyTorch — neural network training & inference
Librosa — audio feature extraction
NumPy / SciPy — signal processing
Scikit-learn — dataset splitting
GitHub — version control & PR workflow

🎯 Learning Outcomes
Audio feature engineering (mel, chroma)
CNN design for time–frequency data
Music theory applied to AI systems
Real-world ML training & inference pipeline
Professional GitHub workflow (branches, PRs)

🎹 MIDI chord export for GarageBand
🌐 YouTube → audio ingestion
🧠 Ensemble (template + neural fusion)
📈 Training on large datasets (GTZAN, GiantSteps)
🎼 Sheet music generation (MusicXML)
📜 License
MIT License
