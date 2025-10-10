# recommender/models/base_model.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class BaseModel(ABC):
    """
    Abstract base class for all recommendation models.

    Every subclass must implement:
        - fit(): train or prepare the model
        - recommend(): return top-N recommendations for a user
    """

    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Train or initialize the model on provided data.

        Args:
            data: Model-specific training data
        """
        pass

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Generate top-N item recommendations for a given user.

        Args:
            user_id: User ID (as row index or mapped integer)
            top_k: Number of recommendations to return

        Returns:
            A list of (item_id, score) tuples, sorted by score descending.
        """
        pass
