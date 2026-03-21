from ml.scripts.dataset import load_manifest, KeyDataset

items = load_manifest("ml/data/manifest.tsv", root_dir=None)
ds = KeyDataset(items, training=True)

X, y = ds[0]
print("X shape:", X.shape)  # expected: (1, 128, frames)
print("y class:", y)        # expected: 0..23

