import tables
import json
import sys
sys.path.append('./MSongsDB/PythonSrc')  # Replace with actual path
import hdf5_getters

"""
testing one file and print out example attributes of a song 
"""
# note that each TR... file is a track which often contain only one song, and occassionally more than one songs (not sure)
# the MillionSongSubset and lastfm_subset are in the same structure fortunately 
testfile = "./DATASET/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5"
testfile_lfm = "./DATASET/lastfm_subset/A/A/A/TRAAAAW128F429D538.json"

h5 = hdf5_getters.open_h5_file_read(testfile)
duration = hdf5_getters.get_duration(h5)
# timbre is in <class 'numpy.ndarray'>
timbre = hdf5_getters.get_segments_timbre(h5)
similar_artists = hdf5_getters.get_similar_artists(h5)
artist_id = hdf5_getters.get_artist_id(h5)
artist_familiarity = hdf5_getters.get_artist_familiarity(h5)
h5.close()

print("duration", duration)
# (971, 12) 
print("timbre", timbre.shape)
print("similar_artists", similar_artists)
print("artist_id", artist_id)
print("artist_familiarity", artist_familiarity)


def parse_lastfm_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract key similarity data
    similar_songs = {
        'track_ids': [sim[0] for sim in data.get('similars', [])],
        'similarity_scores': [sim[1] for sim in data.get('similars', [])]
    }
    
    return {
        'track_id': data.get('track_id'),
        'title': data.get('title'),
        'artist': data.get('artist'),
        'tags': [(tag[0], int(tag[1])) for tag in data.get('tags', [])],
        'similar_songs': similar_songs,
        'timestamp': data.get('timestamp')
    }

def analyze_song_pair(h5_path, json_path):
    # Get MSD audio features
    h5 = hdf5_getters.open_h5_file_read(h5_path)
    features = {
        'timbre': hdf5_getters.get_segments_timbre(h5),
        'duration': hdf5_getters.get_duration(h5),
        'artist_id': hdf5_getters.get_artist_id(h5).decode('utf-8')
    }
    h5.close()
    
    # Get Last.fm metadata
    lastfm_data = parse_lastfm_json(json_path)
    
    # Combined analysis
    return {
        **features,
        **lastfm_data,
        'num_similar_songs': len(lastfm_data['similar_songs']['track_ids']),
        'avg_similarity': sum(lastfm_data['similar_songs']['similarity_scores']) / 
                         max(1, len(lastfm_data['similar_songs']['similarity_scores']))
    }


result = analyze_song_pair(testfile, testfile_lfm)
print(f"Artist: {result['artist']}")
print(f"Title: {result['title']}")
print(f"Top tag: {result['tags'][0][0]} ({result['tags'][0][1]} strength)")
print(f"Most similar song: {result['similar_songs']['track_ids'][0]} "
      f"(score: {result['similar_songs']['similarity_scores'][0]:.2f})")
print(f"Timbre shape: {result['timbre'].shape}")
print(f"num_similar_songs: {result['num_similar_songs']}")
print(f"avg_similarity: {result['avg_similarity']}")


