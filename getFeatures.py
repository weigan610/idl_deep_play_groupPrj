import os
import json
import h5py
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./MSongsDB/PythonSrc')  # Replace with actual path
import hdf5_getters

def get_all_h5_paths(root_dir):
    """Fetch all .h5 files under MillionSongSubset"""
    h5_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if dirpath.count(os.sep) - root_dir.count(os.sep) == 3:  # 4th level
            for f in filenames:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(dirpath, f))
    return h5_files

def get_corresponding_json(h5_path):
    """Convert h5 path to corresponding json path"""
    parts = h5_path.split(os.sep)
    # Replace 'MillionSongSubset' with 'lastfm_subset' and change extension
    json_path = os.sep.join(parts[:-5] + ['lastfm_subset'] + parts[-4:-1] + 
                [parts[-1].replace('.h5', '.json')])
    return json_path if os.path.exists(json_path) else None

def extract_features(h5_path):
    """Extract audio features using the correct MSD getters"""
    h5 = hdf5_getters.open_h5_file_read(h5_path)
    try:
        features = {
            'timbre': hdf5_getters.get_segments_timbre(h5),
            'chroma': hdf5_getters.get_segments_pitches(h5),
            'loudness': hdf5_getters.get_segments_loudness_max(h5)[:, None],
            'duration': hdf5_getters.get_duration(h5),
            'track_id': hdf5_getters.get_track_id(h5).decode('utf-8')
        }
    finally:
        h5.close()
    return features

def create_feature_diff(anchor_feat, other_feat):
    """Create difference features between two songs"""
    # Dynamic time warping for temporal alignment would go here
    # For simplicity, we'll just take mean differences
    return {
        'timbre_diff': anchor_feat['timbre'].mean(0) - other_feat['timbre'].mean(0),
        'chroma_diff': anchor_feat['chroma'].mean(0) - other_feat['chroma'].mean(0),
        'loudness_diff': anchor_feat['loudness'].mean() - other_feat['loudness'].mean(),
        'duration_diff': anchor_feat['duration'] - other_feat['duration']
    }

def process_dataset(h5_root, output_file='song_pairs.npz'):
    h5_files = get_all_h5_paths(os.path.join(h5_root, 'MillionSongSubset'))
    all_pairs = []
    
    count = 0
    for h5_path in tqdm(h5_files, desc='Processing songs'):
        count += 1
        if count > 10:
            break
        json_path = get_corresponding_json(h5_path)
        if not json_path:
            continue
            
        try:
            # Load anchor features
            anchor_feat = extract_features(h5_path)
            
            # Load similarity pairs from JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            for similar_track_id, similarity in data.get('similars', []):
                if similar_track_id == anchor_feat['track_id']:
                    continue  # Skip self-comparison
                    
                # Find the paired h5 file
                similar_h5 = find_h5_from_id(h5_root, similar_track_id)
                if not similar_h5:
                    continue
                    
                # Extract features from paired song
                other_feat = extract_features(similar_h5)
                
                # Create difference features
                feature_diff = create_feature_diff(anchor_feat, other_feat)
                
                all_pairs.append({
                    'anchor_id': anchor_feat['track_id'],
                    'other_id': similar_track_id,
                    'features': feature_diff,
                    'similarity': float(similarity)
                })
        except Exception as e:
            print(f"Skipping {h5_path} due to error: {str(e)}")
            continue
    
    # Save results
    np.savez_compressed(
        output_file,
        anchor_ids=[p['anchor_id'] for p in all_pairs],
        other_ids=[p['other_id'] for p in all_pairs],
        timbre_diffs=np.stack([p['features']['timbre_diff'] for p in all_pairs]),
        chroma_diffs=np.stack([p['features']['chroma_diff'] for p in all_pairs]),
        loudness_diffs=np.array([p['features']['loudness_diff'] for p in all_pairs]),
        duration_diffs=np.array([p['features']['duration_diff'] for p in all_pairs]),
        similarities=np.array([p['similarity'] for p in all_pairs])
    )
    print(f"Saved {len(all_pairs)} pairs to {output_file}")

def find_h5_from_id(root_dir, track_id):
    """Find h5 file path given a track ID"""
    # Example: TRAAAAW128F429D538 -> A/A/A/TRAAAAW128F429D538.h5
    subpath = os.path.join(track_id[2], track_id[3], track_id[4], f"{track_id}.h5")
    full_path = os.path.join(root_dir, 'MillionSongSubset', subpath)
    return full_path if os.path.exists(full_path) else None

process_dataset('./DATASET')


