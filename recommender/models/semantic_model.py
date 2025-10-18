# recommender/models/semantic_model.py

from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseModel


class SemanticModel(BaseModel):
    """
    Transformer-based semantic recommender.

    Each artist's tags (or textual metadata) are embedded into a dense vector
    using a pretrained transformer model (SentenceTransformer). Similarity
    between artists is computed via cosine similarity in embedding space.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the semantic model.

        Args:
            model_name: Hugging Face model name to load.
            device: 'cpu' or 'cuda'; if None, auto-detect.
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        self.artist_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Core training / inference methods
    # ------------------------------------------------------------
    def fit(self, tag_corpus: Dict[int, str]) -> None:
        """
        Encode all artist tag texts into dense embeddings.

        Args:
            tag_corpus: {artist_id: "tag1 tag2 tag3 ..."}.
        """
        self.artist_ids = list(tag_corpus.keys())
        tag_texts = list(tag_corpus.values())

        print(f"Encoding {len(tag_texts)} artists using {self.model_name} ...")
        emb = self.model.encode(
            tag_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        self.embeddings = np.asarray(emb, dtype=np.float32)
        print("Encoding complete. Embedding shape:", self.embeddings.shape)

    def recommend(
        self, artist_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-N similar artists based on semantic embedding similarity.

        Args:
            artist_id: Target artist ID.
            top_k: Number of results to return.

        Returns:
            List of (artist_id, cosine_similarity_score).
        """
        if self.embeddings is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if artist_id not in self.artist_ids:
            raise ValueError(f"Artist ID {artist_id} not found in corpus.")

        idx = self.artist_ids.index(artist_id)
        target_vec = self.embeddings[idx : idx + 1]

        scores = cosine_similarity(target_vec, self.embeddings).flatten()
        top_idx = np.argsort(scores)[::-1][1 : top_k + 1]  # skip self

        return [
            (self.artist_ids[i], float(scores[i])) for i in top_idx
        ]

    def recommend_for_multiple_seeds(
        self, artist_ids: List[int], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Recommend artists similar to multiple seed artists.

        Args:
            artist_ids: List of seed artist IDs.
            top_k: Number of results to return.

        Returns:
            List of (artist_id, similarity_score) tuples.
        """
        if self.embeddings is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        valid_idx = [self.artist_ids.index(aid) for aid in artist_ids if aid in self.artist_ids]
        if not valid_idx:
            raise ValueError("No valid seed artists found.")

        seed_vec = self.embeddings[valid_idx]
        mean_vec = np.mean(seed_vec, axis=0, keepdims=True)

        scores = cosine_similarity(mean_vec, self.embeddings).flatten()
        top_idx = np.argsort(scores)[::-1][1 : top_k + 1]

        return [
            (self.artist_ids[i], float(scores[i])) for i in top_idx
        ]
