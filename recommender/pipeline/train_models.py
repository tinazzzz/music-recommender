# recommender/pipeline/train_models.py

import os, json, joblib
from recommender.datasets.lastfm_processor import LastFMDataset
from recommender.models.mf_model import MFModel
from recommender.models.semantic_model import SemanticModel

# ======================================================
# Config paths
# ======================================================
DATA_DIR = "data/raw/hetrec2011-lastfm-2k"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "model"
SEMANTIC_MODEL_DIR = os.path.join(MODEL_DIR, "semantic_model")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SEMANTIC_MODEL_DIR, exist_ok=True)

# ======================================================
# Step 1 — Preprocess
# ======================================================
print("📦 Loading and preprocessing dataset...")
dataset = LastFMDataset(DATA_DIR)
dataset.load()
processed = dataset.preprocess()

# Save processed corpus for reproducibility
tag_corpus_path = os.path.join(PROCESSED_DIR, "tag_corpus.pkl")
joblib.dump(processed["tag_corpus"], tag_corpus_path)
print(f"✅ Saved tag corpus → {tag_corpus_path}")

# ======================================================
# Step 2 — Train MF
# ======================================================
print("\n🎵 Training MF model...")
mf = MFModel(factors=64, iterations=20, regularization=0.01)
mf.fit(processed)
joblib.dump(mf, os.path.join(MODEL_DIR, "mf_model.pkl"))
print(f"✅ Saved MF model → {MODEL_DIR}/mf_model.pkl")

# ======================================================
# Step 3 — Train Semantic Transformer
# ======================================================
print("\n🧠 Training SemanticModel...")
sem = SemanticModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
sem.fit(processed["tag_corpus"])

# Save transformer weights (the model)
sem.model.save(SEMANTIC_MODEL_DIR)
print(f"✅ Saved transformer weights → {SEMANTIC_MODEL_DIR}")

# Save embeddings & metadata separately under processed/
embeddings_path = os.path.join(PROCESSED_DIR, "semantic_embeddings.npy")
metadata_path = os.path.join(PROCESSED_DIR, "semantic_metadata.json")

joblib.dump(sem.embeddings, embeddings_path)
json.dump({"artist_ids": sem.artist_ids}, open(metadata_path, "w"))
print(f"✅ Saved embeddings → {embeddings_path}")
print(f"✅ Saved metadata → {metadata_path}")

print("\n🎉 All artifacts saved successfully!")
