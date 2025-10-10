# recommender/datasets/base_processor.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from scipy.sparse import csr_matrix


class BaseDataset(ABC):
    """
    Abstract base class for dataset loaders.
    Provides tools for loading, preprocessing, and converting user item interactions into sparse matrices.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize the dataset base.

        Args:
            data_dir: Path to the dataset folder.
        """
        self.data_dir: str = data_dir
        self.data: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def load(self) -> None:
        """
        Load raw data files into memory as pandas DataFrames.
        Must populate self.data with relevant keys.
        """
        pass

    @abstractmethod
    def preprocess(self) -> Dict[str, Any]:
        """
        Transform raw data into model-ready structures (matrices, mappings).

        Returns:
            A dictionary containing standardized dataset components.
        """
        pass

    def get_user_item_matrix(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: Optional[str] = None,
    ) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
        """
        Convert a long-form user item DataFrame into a sparse CSR matrix.

        Args:
            df: Input DataFrame containing interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.
            value_col (optional): Column name for interaction strength.

        Returns:
            A tuple of:
                - csr_matrix: user item interaction matrix
                - user_map: mapping of row indices → user IDs
                - item_map: mapping of column indices → item IDs
        """
        
        users = df[user_col].astype("category")
        items = df[item_col].astype("category")

        user_map: Dict[int, int] = dict(enumerate(users.cat.categories))
        item_map: Dict[int, int] = dict(enumerate(items.cat.categories))

        values = df[value_col].values if value_col else [1] * len(df)

        mat = csr_matrix(
            (values, (users.cat.codes, items.cat.codes)),
            shape=(len(user_map), len(item_map))
        )

        return mat, user_map, item_map
