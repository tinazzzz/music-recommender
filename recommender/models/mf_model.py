# recommender/models/mf_model.py

from typing import Dict, List, Tuple, Any
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from .base_model import BaseModel
import numpy as np


class MFModel(BaseModel):
    """
    Matrix Factorization model using Alternating Least Squares (ALS)
    for implicit feedback recommendation.

    Attributes:
        model: ALS instance from the `implicit` library
        user_map: Mapping of matrix row indices to user IDs
        item_map: Mapping of matrix column indices to item IDs
    """

    def __init__(
        self,
        factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        use_gpu: bool = False
    ) -> None:
        """
        Initialize the MF model configuration.

        Args:
            factors: Latent dimension size.
            regularization: Regularization term for ALS optimization.
            iterations: Number of ALS training iterations.
            use_gpu: Whether to use GPU acceleration (if available).
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
        )
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self.R: csr_matrix | None = None

    def fit(self, data: Dict[str, Any]) -> None:
        """
        Train the ALS model on a user-item interaction matrix.

        Args:
            data: Dictionary from LastFMDataset.preprocess() with keys:
                - "matrix": csr_matrix of user-item interactions
                - "user_map": Dict[int, int]
                - "artist_map": Dict[int, int]
        """
        R: csr_matrix = data["matrix"]
        self.user_map = data["user_map"]
        self.item_map = data["artist_map"]
        self.R = R

        # ALS expects item–user matrix for training
        self.model.fit(R.T)

    def recommend(
        self,
        user_id: int | None = None,
        user_vector: np.ndarray | None = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
    """
    Recommend items for a known or temporary user.

    Args:
        user_id: Known user row index (for existing users).
        user_vector: Computed embedding (for new users).
        top_k: Number of recommendations.

    Returns:
        List of (artist_id, score) tuples.
    """
    if self.R is None:
        raise ValueError("Model must be trained before recommending.")

    if user_vector is not None:
        # Cold-start user (fold-in)
        scores = self.model.item_factors @ user_vector
        top_items = np.argsort(scores)[::-1][:top_k]
        inv_item_map = {v: k for k, v in self.item_map.items()}
        return [(inv_item_map[i], float(scores[i])) for i in top_items]

    elif user_id is not None:
        # Existing user
        recs = self.model.recommend(user_id, self.R, N=top_k)
        inv_item_map = {v: k for k, v in self.item_map.items()}
        return [(inv_item_map[i], float(s)) for i, s in recs]

    else:
        raise ValueError("Either user_id or user_vector must be provided.")


    def get_user_vector(
        self,
        user_interactions: Dict[int, float],
        reg: float = 0.1
    ) -> np.ndarray:
        """
        Compute a temporary embedding for a new user via folding-in.

        Args:
            user_interactions: Dict[item_id, weight or rating].
            reg: Regularization parameter for ridge regression.

        Returns:
            Numpy array (1 x latent_factors) representing the user vector.
        """
        if self.R is None:
            raise ValueError("Model must be trained before inferring new user vectors.")

        item_factors = self.model.item_factors  # shape: (n_items, k)
        item_map_inv = {v: k for k, v in self.item_map.items()}

        # Filter to known items
        item_ids = [item_map_inv[i] for i in user_interactions.keys() if i in item_map_inv]
        if not item_ids:
            raise ValueError("No known items found for projection.")

        ratings = np.array(list(user_interactions.values()), dtype=np.float64)
        V = item_factors[item_ids, :]  # shape (m, k)

        A = V.T @ V + reg * np.eye(V.shape[1])
        b = V.T @ ratings
        user_vec = np.linalg.solve(A, b)
        return user_vec
        