import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append('./MSongsDB/PythonSrc')  # Replace with actual path
import hdf5_getters

class H5SimilarityDataset(Dataset):
    def __init__(self, h5_root_dir, json_root_dir, lyrics_file):
        """
        Args:
            h5_root_dir: Root directory of MillionSongSubset
            json_root_dir: Root directory of lastfm_subset
        """
        self.h5_root_dir = h5_root_dir
        self.json_root_dir = json_root_dir
        self.lyrics_file = lyrics_file
        # First build track_id to path mapping
        self.h5_path_map = self._build_h5_path_map(h5_root_dir)
        self.json_path_map = self._build_json_path_map(json_root_dir)
        
        # Then build list of valid (anchor, similar) pairs
        self.pairs = self._build_similarity_pairs()
        self.lyrics_dict = self._get_lyrics()

    def _build_h5_path_map(self, root_dir):
        """Create {track_id: h5_path} mapping"""
        path_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.h5'):
                    track_id = f.split('.')[0]
                    path_map[track_id] = os.path.join(dirpath, f)
        return path_map

    def _build_json_path_map(self, root_dir):
        """Create {track_id: json_path} mapping"""
        path_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.json'):
                    track_id = f.split('.')[0]
                    path_map[track_id] = os.path.join(dirpath, f)
        return path_map

    def _build_similarity_pairs(self):
        """Create list of (anchor_id, similar_id, similarity) tuples"""
        pairs = []
        for anchor_id, json_path in tqdm(self.json_path_map.items(), desc="Processing JSON files"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                for similar_id, similarity in data.get('similars', []):
                    if similar_id in self.h5_path_map:  # Only include if we have both files
                        pairs.append((anchor_id, similar_id, float(similarity)))
            except Exception as e:
                print(f"Skipping {json_path} due to error: {str(e)}")
                continue
        return pairs

    def _get_features(self, track_id):
        """Extract features for a single track"""
        h5_path = self.h5_path_map[track_id]
        h5 = hdf5_getters.open_h5_file_read(h5_path)
        
        try:
            features = {
                'timbre': hdf5_getters.get_segments_timbre(h5),
                'chroma': hdf5_getters.get_segments_pitches(h5),
                # 'loudness': hdf5_getters.get_segments_loudness_max(h5)[:, None],
                # 'duration': hdf5_getters.get_duration(h5)
            }
        finally:
            h5.close()
        return features
    
    def _get_lyrics(self):
        print("Loading musiXmatch lyrics...")

        with open(self.lyrics_file, encoding='utf-8') as f:
            # lines = f.readlines()
            lines = f.readlines()[:20]

        # Parse each song's BOW vector
        lyrics_dict = {}
        for i in range(1, len(lines)):  # start from 1 to skip header
            line = lines[i]
            parts = line.strip().split(',', 1)  # <--- important: only split at first comma!

            if len(parts) < 2:
                continue  # skip broken lines

            track_id = parts[0]
            lyrics = parts[1]

            bow = {}

            # Split lyrics into words
            words = lyrics.strip().split()
            for word in words:
                bow[word] = bow.get(word, 0) + 1

            lyrics_dict[track_id] = bow

        return lyrics_dict


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_id, similar_id, similarity = self.pairs[idx]
        
        # Get features for both tracks
        anchor_feat = self._get_features(anchor_id)
        similar_feat = self._get_features(similar_id)
        anchor_lyrics = self.lyrics_dict.get(anchor_id, {})
        similar_lyrics = self.lyrics_dict.get(similar_id, {})
        
        anchor_feat = torch.cat([
            torch.FloatTensor(anchor_feat['timbre']),
            torch.FloatTensor(anchor_feat['chroma']),
        ])
        similar_feat = torch.cat([
            torch.FloatTensor(similar_feat['timbre']),
            torch.FloatTensor(similar_feat['chroma']),
        ])
        
        label = torch.FloatTensor([similarity])
        
        return anchor_feat, similar_feat, anchor_lyrics, similar_lyrics,label
    
    # Add this method to your original H5SimilarityDataset class
    def __getitem_from_pair__(self, pair):
        anchor_id, similar_id, similarity = pair
        similarity = float(similarity)
        feature1 = self._get_features(anchor_id)
        feature2 = self._get_features(similar_id)
        anchor_lyrics = self.lyrics_dict.get(anchor_id, {})
        similar_lyrics = self.lyrics_dict.get(similar_id, {})

        feature1 = torch.cat([
            torch.FloatTensor(feature1['timbre']),
            torch.FloatTensor(feature1['chroma']),
        ])
        feature2 = torch.cat([
            torch.FloatTensor(feature2['timbre']),
            torch.FloatTensor(feature2['chroma']),
        ])

        label = torch.FloatTensor([similarity])
        return feature1, feature2, anchor_lyrics, similar_lyrics, label
    
    def collate_fn(batch):
        """
        Collate function that handles variable-length sequences by padding to max length in batch
        Args:
            batch: List of tuples (anchor_feat, similar_feat, label)
        Returns:
            anchor_feats: Padded tensor (batch_size, max_len, 24)
            similar_feats: Padded tensor (batch_size, max_len, 24)
            labels: Tensor (batch_size, 1)
            lengths: Tuple of (anchor_lengths, similar_lengths)
        """
        anchors, similars, anchor_lyrics, similar_lyrics, labels = zip(*batch)

        labels = torch.stack(labels, dim=0)

        def pad_sequences(features):
            lengths = [f.shape[0] for f in features]
            max_len = max(lengths)
            feature_dim = features[0].shape[1]

            padded = torch.zeros(len(features), max_len, feature_dim)
            for i, (seq, seq_len) in enumerate(zip(features, lengths)):
                padded[i, :seq_len] = seq

            return padded, torch.tensor(lengths)

        anchor_feats, anchor_lengths = pad_sequences(anchors)
        similar_feats, similar_lengths = pad_sequences(similars)

        return anchor_feats, similar_feats, anchor_lyrics, similar_lyrics, labels, anchor_lengths, similar_lengths

class H5TestDataset(Dataset):
    def __init__(self, h5_root_dir, json_root_dir):
        """
        Args:
            h5_root_dir: Root directory of MillionSongSubset
            json_root_dir: Root directory of lastfm_subset
        """
        self.h5_root_dir = h5_root_dir
        self.json_root_dir = json_root_dir
        
        self.h5_path_map = self._build_h5_path_map(h5_root_dir)
        self.json_path_map = self._build_json_path_map(json_root_dir)
        
        self.pairs = self._build_similarity_pairs()
        self.lyrics_dict = self._get_lyrics()

    def _build_h5_path_map(self, root_dir):
        """Create {track_id: h5_path} mapping"""
        path_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.h5'):
                    track_id = f.split('.')[0]
                    path_map[track_id] = os.path.join(dirpath, f)
        return path_map

    def _build_json_path_map(self, root_dir):
        """Create {track_id: json_path} mapping"""
        path_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.json'):
                    track_id = f.split('.')[0]
                    path_map[track_id] = os.path.join(dirpath, f)
        return path_map

    def _build_similarity_pairs(self):
        """Create list of (anchor_id, similar_id, similarity) tuples"""
        pairs = []
        for anchor_id, json_path in tqdm(self.json_path_map.items(), desc="Processing JSON files"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                for similar_id, _ in data.get('similars', []):
                    if similar_id in self.h5_path_map:  # Only include if we have both files
                        pairs.append((anchor_id, similar_id))
            except Exception as e:
                print(f"Skipping {json_path} due to error: {str(e)}")
                continue
        return pairs
    
    def _get_lyrics(self):
        print("Loading musiXmatch lyrics...")

        with open(self.lyrics_file, encoding='utf-8') as f:
            # lines = f.readlines()
            lines = f.readlines()[:20]

        # Parse each song's BOW vector
        lyrics_dict = {}
        for i in range(1, len(lines)):  # start from 1 to skip header
            line = lines[i]
            parts = line.strip().split(',', 1)  # <--- important: only split at first comma!

            if len(parts) < 2:
                continue  # skip broken lines

            track_id = parts[0]
            lyrics = parts[1]

            bow = {}

            # Split lyrics into words
            words = lyrics.strip().split()
            for word in words:
                bow[word] = bow.get(word, 0) + 1

            lyrics_dict[track_id] = bow

        return lyrics_dict

    def _get_features(self, track_id):
        """Extract features for a single track"""
        h5_path = self.h5_path_map[track_id]
        h5 = hdf5_getters.open_h5_file_read(h5_path)
        try:
            features = {
                'timbre': hdf5_getters.get_segments_timbre(h5),
                'chroma': hdf5_getters.get_segments_pitches(h5),
                # 'loudness': hdf5_getters.get_segments_loudness_max(h5)[:, None],
                # 'duration': hdf5_getters.get_duration(h5)
            }
        finally:
            h5.close()
        return features

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_id, similar_id = self.pairs[idx]
        
        # Get features for both tracks
        anchor_feat = self._get_features(anchor_id)
        similar_feat = self._get_features(similar_id)
        
        anchor_feat = torch.cat([
            torch.FloatTensor(anchor_feat['timbre']),
            torch.FloatTensor(anchor_feat['chroma']),
        ])
        similar_feat = torch.cat([
            torch.FloatTensor(similar_feat['timbre']),
            torch.FloatTensor(similar_feat['chroma']),
        ])
        
        return anchor_feat, similar_feat
    
    def __getitem_from_pair__(self, pair):
        anchor_id, similar_id = pair
        feature1 = self._get_features(anchor_id)
        feature2 = self._get_features(similar_id)
        anchor_lyrics = self.lyrics_dict.get(anchor_id, {})
        similar_lyrics = self.lyrics_dict.get(similar_id, {})

        feature1 = torch.cat([
            torch.FloatTensor(feature1['timbre']),
            torch.FloatTensor(feature1['chroma']),
        ])
        feature2 = torch.cat([
            torch.FloatTensor(feature2['timbre']),
            torch.FloatTensor(feature2['chroma']),
        ])

        return feature1, feature2, anchor_lyrics, similar_lyrics
    
    def collate_fn(batch):
        """
        Collate function that handles variable-length sequences by padding to max length in batch
        Args:
            batch: List of tuples (anchor_feat, similar_feat, label)
        Returns:
            anchor_feats: Padded tensor (batch_size, max_len, 24)
            similar_feats: Padded tensor (batch_size, max_len, 24)
            labels: Tensor (batch_size, 1)
            lengths: Tuple of (anchor_lengths, similar_lengths)
        """
        anchors, similars, anchor_lyrics, similar_lyrics, _ = zip(*batch)


        def pad_sequences(features):
            lengths = [f.shape[0] for f in features]
            max_len = max(lengths)
            feature_dim = features[0].shape[1]

            padded = torch.zeros(len(features), max_len, feature_dim)
            for i, (seq, seq_len) in enumerate(zip(features, lengths)):
                padded[i, :seq_len] = seq

            return padded, torch.tensor(lengths)

        anchor_feats, anchor_lengths = pad_sequences(anchors)
        similar_feats, similar_lengths = pad_sequences(similars)

        return anchor_feats, similar_feats, anchor_lyrics, similar_lyrics, anchor_lengths, similar_lengths

def get_splits(dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train/val/test while preserving pairs
    Returns: (train_dataset, val_dataset, test_dataset)
    """
    # Get all pairs and convert to numpy array for indexing
    all_pairs = np.array(dataset.pairs)
    
    # First split into train+val and test
    train_val_pairs, test_pairs = train_test_split(
        all_pairs, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Then split train+val into train and val
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, 
        test_size=val_size/(1-test_size), 
        random_state=random_state
    )
    
    # Create subset datasets
    train_dataset = PairSubsetDataset(dataset, train_pairs)
    val_dataset = PairSubsetDataset(dataset, val_pairs)
    test_dataset = PairSubsetDataset(dataset, test_pairs)
    
    return train_dataset, val_dataset, test_dataset

class PairSubsetDataset(Dataset):
    """Wrapper to create subsets of the main dataset"""
    def __init__(self, original_dataset, pairs):
        self.original_dataset = original_dataset
        self.pairs = pairs.tolist() if isinstance(pairs, np.ndarray) else pairs
        
        # Copy necessary attributes
        self.h5_path_map = original_dataset.h5_path_map
        self.json_path_map = original_dataset.json_path_map
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # Delegate to original dataset with the subset pair
        return self.original_dataset.__getitem_from_pair__(self.pairs[idx])
    
