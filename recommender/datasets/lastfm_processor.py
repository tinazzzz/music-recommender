# recommender/datasets/lastfm_processor.py

import os
import pandas as pd
from typing import Dict, Any, Tuple
from .base_processor import BaseDataset


class LastFMDataset(BaseDataset):
    """
    Loader for the HetRec 2011 Last.fm dataset.

    Produces:
        - User-Artist interaction matrix for MF
        - Artist-Tag corpus for semantic search
        - Metadata for future models
    """

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)

    def load(self) -> None:
        """
        Load the HetRec 2011 Last.fm dataset files into memory.
        Expected files:
            - artists.dat
            - user_artists.dat
            - tags.dat
            - user_taggedartists.dat
            - (optional) user_friends.dat
            - (optional) user_taggedartists-timestamps.dat

        """
        def load_file(filename: str) -> pd.DataFrame:
            path = os.path.join(self.data_dir, f"{filename}.dat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")
            return pd.read_csv(path, sep="\t")

        # Required files
        self.data["artists"] = load_file("artists")
        self.data["user_artists"] = load_file("user_artists")
        self.data["tags"] = load_file("tags")
        self.data["user_taggedartists"] = load_file("user_taggedartists")

        # Optional files
        if os.path.exists(os.path.join(self.data_dir, "user_friends.dat")):
            self.data["user_friends"] = load_file("user_friends")
        if os.path.exists(os.path.join(self.data_dir, "user_taggedartists-timestamps.dat")):
            self.data["user_taggedartists_timestamps"] = load_file("user_taggedartists-timestamps")

    def preprocess(self) -> Dict[str, Any]:
        """
        Preprocess dataset for multiple recommender paradigms.

        Returns:
        Dict with:
                - matrix: csr_matrix (user-artist playcounts)
                - user_map: Dict[int, int]
                - artist_map: Dict[int, int]
                - tag_corpus: Dict[int, str]
                - artists_meta: pd.DataFrame

        """
        # User-artist matrix (for MF) 
        interactions = self.data["user_artists"].copy()
        interactions["weight"] = interactions["weight"] / interactions["weight"].max()

        matrix, user_map, artist_map = self.get_user_item_matrix(
            df=interactions,
            user_col="userID",
            item_col="artistID",
            value_col="weight"
        )

        # Artist-tag corpus (for semantic model) 
        tags = self.data["tags"]
        artist_tags = self.data["user_taggedartists"].merge(tags, on="tagID", how="left")
        artist_tags["tagValue"] = artist_tags["tagValue"].astype(str)
        tag_corpus: Dict[int, str] = (
            artist_tags.groupby("artistID")["tagValue"]
            .apply(lambda x: " ".join(x))
            .to_dict()
        )

        # Artist metadata (for display or future use)
        artists_meta = (
            self.data["artists"][["id", "name", "url", "pictureURL"]]
            .rename(columns={"id": "artistID"})
        )

        return {
            "matrix": matrix,
            "user_map": user_map,
            "artist_map": artist_map,
            "tag_corpus": tag_corpus,
            "artists_meta": artists_meta,
        }

    def print(self) -> None:
        """
        Print dataset statistics for verification.
        """
        print("=== Last.fm HetRec 2011 Dataset ===")
        print(f"Artists: {len(self.data.get('artists', []))}")
        print(f"Users (unique): {self.data['user_artists']['userID'].nunique()}")
        print(f"Interactions: {len(self.data['user_artists'])}")
        print(f"Tags: {len(self.data.get('tags', []))}")
        print(f"Tagged artists: {len(self.data.get('user_taggedartists', []))}")
        print(f"Friends (if loaded): {len(self.data.get('user_friends', []))}")
