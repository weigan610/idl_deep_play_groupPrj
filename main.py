import torch
from dataloader import H5SimilarityDataset, H5TestDataset, get_splits
from torch.utils.data import Dataset, DataLoader
from model import CrossSimilarityModel
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
from model import Lyrics_Encoder, SiameseNetwork, SimilarityDecoder
import torch.nn.functional as F

dataset = H5SimilarityDataset(
    h5_root_dir='./DATASET/MillionSongSubset',
    json_root_dir='./DATASET/lastfm_subset',
    lyrics_file = 'track_lyrics.csv'
)


train_dataset, val_dataset, test_dataset = get_splits(dataset)
print("train_dataset, val_dataset, test_dataset sizes:", 
      len(train_dataset.pairs),
      len(val_dataset.pairs),
      len(test_dataset.pairs),
      )

print(len(train_dataset.pairs[0]))
# anchor_id
print(train_dataset.pairs[0][0])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,          # Parallel loading
    pin_memory=True,        # Faster transfer to GPU
    collate_fn=H5SimilarityDataset.collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,          # Parallel loading
    pin_memory=True,        # Faster transfer to GPU
    collate_fn=H5SimilarityDataset.collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,          # Parallel loading
    pin_memory=True,        # Faster transfer to GPU
    collate_fn=H5TestDataset.collate_fn
)

# sanity check
for batch in train_loader:
    anchor_feats, similar_feats, anchor_lyrics, similar_lyrics, labels, anchor_lengths, similar_lengths = batch
    
    print("\nBatch shapes:")
    print(f"anchor_feats: {anchor_feats.shape}")  # (batch_size, max_len, 24)
    print(f"similar_feats: {similar_feats.shape}")  # (batch_size, max_len, 24)
    print(f"labels: {labels.shape}")  # (batch_size, 1)
    print(f"lyrics1: {anchor_lyrics[1]}")  # (batch_size, max_len, 24)
    print(f"lyrics2: {similar_lyrics[1]}")  # (batch_size, 1)
    
    print("\nSequence lengths:")
    print(f"anchor_lengths: {len(anchor_lengths)} items - sample lengths: {anchor_lengths[:5]}...")  # First 5 lengths
    print(f"similar_lengths: {len(similar_lengths)} items - sample lengths: {similar_lengths[:5]}...")  # First 5 lengths
    
    # Example showing how to access individual sequence lengths
    print("\nExample sequence access:")
    print(f"First anchor (padded): {anchor_feats[0].shape}")
    print(f"First anchor (actual): {anchor_feats[0, :anchor_lengths[0]].shape}")
    break

for batch in test_loader:
    anchor_feats, similar_feats, anchor_lyrics, similar_lyrics, anchor_lengths, similar_lengths = batch
    
    print("\nBatch shapes:")
    print(f"anchor_feats: {anchor_feats.shape}")  # (batch_size, max_len, 24)
    print(f"similar_feats: {similar_feats.shape}")  # (batch_size, max_len, 24)
    
    print("\nSequence lengths:")
    print(f"anchor_lengths: {len(anchor_lengths)} items - sample lengths: {anchor_lengths[:5]}...")  # First 5 lengths
    print(f"similar_lengths: {len(similar_lengths)} items - sample lengths: {similar_lengths[:5]}...")  # First 5 lengths
    
    # Example showing how to access individual sequence lengths
    print("\nExample sequence access:")
    print(f"First anchor (padded): {anchor_feats[0].shape}")
    print(f"First anchor (actual): {anchor_feats[0, :anchor_lengths[0]].shape}")
    break

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQ_LEN = 300
DROPOUT = 0.3
GLOVE_PATH = "glove.6B.100d.txt"

# vocab_size = 


# Assuming you have initialized these somewhere:
lyrics_encoder = Lyrics_Encoder(vocab_size, embedding_dim, hidden_dim)
audio_encoder = SiameseNetwork(embedding_dim=128)  # The CNN Siamese
similarity_decoder = SimilarityDecoder(input_dim=merged_feature_dim)  # You will define this size properly

for batch in train_loader:
    anchor_feats, similar_feats, anchor_lyrics, similar_lyrics, labels, anchor_lengths, similar_lengths = batch

    # 1. Encode Lyrics
    lyrics_merged = Lyrics_Encoder.encode_and_merge(anchor_lyrics, similar_lyrics, lyrics_encoder)

    # 2. Encode Audio
    audio_merged = audio_encoder(anchor_feats, similar_feats, merge_type='cosine_similarity')
    audio_merged = audio_merged.unsqueeze(-1)  # Make sure it has a feature dimension if needed

    # 3. Normalize both merged outputs individually
    lyrics_merged = F.normalize(lyrics_merged, p=2, dim=-1)
    audio_merged = F.normalize(audio_merged, p=2, dim=-1)

    # 4. Match shapes
    # If lyrics_merged is [batch_size, seq_len, feature_dim], pool over seq_len
    if lyrics_merged.ndim == 3:
        lyrics_merged = torch.mean(lyrics_merged, dim=1)  # Mean pooling across sequence

    # 5. Merge lyric and audio embeddings
    final_merged = torch.cat([lyrics_merged, audio_merged], dim=-1)

    # 6. Pass through the similarity decoder
    similarity_score = similarity_decoder(final_merged)

    # 7. Compute Loss
    loss = clc_loss(similarity_score, labels)  # Your contrastive loss




