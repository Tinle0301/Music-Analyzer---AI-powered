NOTE_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def key_to_class(tonic: str, mode: str) -> int:
    tonic = tonic.strip()
    mode = mode.strip().lower()
    if tonic not in NOTE_NAMES_SHARP:
        raise ValueError(f"Unknown tonic: {tonic}")
    if mode not in ("major", "minor"):
        raise ValueError(f"Mode must be 'major' or 'minor', got: {mode}")
    idx = NOTE_NAMES_SHARP.index(tonic)
    return idx if mode == "major" else 12 + idx

def class_to_key(class_id: int) -> tuple[str, str]:
    if not (0 <= class_id <= 23):
        raise ValueError(f"class_id must be in [0,23], got: {class_id}")
    if class_id < 12:
        return NOTE_NAMES_SHARP[class_id], "major"
    return NOTE_NAMES_SHARP[class_id - 12], "minor"
