# --- 1. Imports ---

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import mse_loss
import re
from tqdm import tqdm


# --- 2. Read the CSV files ---

# # Skip the first row
lyrics_df = pd.read_csv('/kaggle/input/lyrics/lyrics.csv', header=0, names=['track_id', 'lyrics', 'score'])

# audio_df = pd.read_csv('/kaggle/input/audios/audio.csv')  # audio is normal comma-separated

# Step 1: Load all DataFrames
# the addresses are in the format e.g.: https://www.kaggle.com/datasets/weiqianzhang987/audio2k-6k
# audio_parts = [
#     # pd.read_pickle('/kaggle/input/audio-pickle/audio.pkl'),
#     # pd.read_pickle('/kaggle/input/audio2k-6k/audio2k-6k.pkl'),
#     pd.read_pickle('/kaggle/input/audio6k-10k/audio6k-10k.pkl'),
#     # pd.read_pickle('/kaggle/input/audio10k-14k/audio10k-14k.pkl'),
#     # pd.read_pickle('/kaggle/input/audio14k-18k/audio14k-18k.pkl'),
#     # pd.read_pickle('/kaggle/input/audio18k-22k/audio18k-22k.pkl'),
#     # pd.read_pickle('/kaggle/input/audio22k-26k/audio22k-26k.pkl')
# ]

# Step 2: Concatenate them
# audio_df = pd.concat(audio_parts, ignore_index=True)
audio_df = pd.read_pickle('/kaggle/input/audio6k-10k/audio6k-10k.pkl')

# Step 3: Drop duplicates based on 'track_id'
# audio_df = audio_df.drop_duplicates(subset='track_id', keep='first')

# Optional: Reset index
audio_df.reset_index(drop=True, inplace=True)

# Check result
print(f"Merged audio_df shape: {audio_df.shape}")

# Build fast lookup for audio waveforms
audio_dict = {row['track_id']: row['waveform'] for _, row in audio_df.iterrows()}
print(len(audio_dict))



# --- 3. Encode the lyrics ---

# Simple tokenizer: split by spaces
def simple_tokenizer(text):
    return text.strip().split()

# Build vocabulary
vocab = {}
for text in lyrics_df['lyrics']:  # just loop over all lyrics
    for word in simple_tokenizer(text):
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Encode function
def encode(text):
    tokens = simple_tokenizer(text)
    encoded = [vocab.get(token, 0) for token in tokens]  # OOV -> 0
    return torch.tensor(encoded, dtype=torch.long)


# --- 4. Updated Dataset Definition ---

class PLyricsAudioDataset(Dataset):
    def __init__(self, lyrics_df, audio_dict):
        self.samples = []
        for i in tqdm(range(0, len(lyrics_df), 2), desc="Building dataset"):
            row1 = lyrics_df.iloc[i]
            row2 = lyrics_df.iloc[i + 1]

            id1, lyrics1, score1 = row1['track_id'], row1['lyrics'], row1['score']
            id2, lyrics2, score2 = row2['track_id'], row2['lyrics'], row2['score']

            # Sanity check
            if score1 != score2:
                break  # this means that something went off track 

            # Make sure audio exists
            if id1 not in audio_dict or id2 not in audio_dict:
                continue

            audio1 = audio_dict[id1]
            audio2 = audio_dict[id2]

            if audio1 is None or audio2 is None:
                continue
                

            # Encode lyrics
            lyrics1_enc = encode(lyrics1)
            lyrics2_enc = encode(lyrics2)
            # Pad lyrics to same length before subtraction
            max_len_lyrics = max(lyrics1_enc.shape[0], lyrics2_enc.shape[0])
            lyrics1_enc = nn.functional.pad(lyrics1_enc, (0, max_len_lyrics - lyrics1_enc.shape[0]))
            lyrics2_enc = nn.functional.pad(lyrics2_enc, (0, max_len_lyrics - lyrics2_enc.shape[0]))
            lyrics_feat = torch.abs(lyrics1_enc.float() - lyrics2_enc.float())
            # Prepare audio features
            audio1_tensor = torch.tensor(audio1, dtype=torch.float32)
            
            audio2_tensor = torch.tensor(audio2, dtype=torch.float32)
            max_len_audio = max(audio1_tensor.shape[0], audio2_tensor.shape[0])
            audio1_tensor = nn.functional.pad(audio1_tensor, (0, max_len_audio - audio1_tensor.shape[0]))
            audio2_tensor = nn.functional.pad(audio2_tensor, (0, max_len_audio - audio2_tensor.shape[0]))
            audio_feat = torch.abs(audio1_tensor - audio2_tensor)
            # Pad to equal final length for stacking
            final_len = max(lyrics_feat.shape[0], audio_feat.shape[0])
            lyrics_feat = nn.functional.pad(lyrics_feat, (0, final_len - lyrics_feat.shape[0]))
            audio_feat = nn.functional.pad(audio_feat, (0, final_len - audio_feat.shape[0]))
            
            # Stack as 2 channels
            feat = torch.stack([lyrics_feat, audio_feat], dim=0)  # shape [2, final_len]

            self.samples.append((feat, torch.tensor(score1, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(batch):
        feats, scores = zip(*batch)  # unpack
        max_len = max(feat.shape[1] for feat in feats)  # find max length in this batch
    
        padded_feats = []
        for feat in feats:
            pad_width = max_len - feat.shape[1]
            if pad_width > 0:
                feat = nn.functional.pad(feat, (0, pad_width))  # pad only last dimension
            padded_feats.append(feat)
    
        feats_tensor = torch.stack(padded_feats)  # now safe to stack
        scores_tensor = torch.stack(scores)
    
        return feats_tensor, scores_tensor
    

class LyricsAudioDataset(Dataset):
    def __init__(self, lyrics_df, audio_dict):
        self.samples = []
        for i in tqdm(range(0, len(lyrics_df), 2), desc="Building dataset"):
            row1 = lyrics_df.iloc[i]
            row2 = lyrics_df.iloc[i + 1]

            id1, lyrics1, score1 = row1['track_id'], row1['lyrics'], row1['score']
            id2, lyrics2, score2 = row2['track_id'], row2['lyrics'], row2['score']

            # Sanity check
            if score1 != score2:
                break  # this means that something went off track 

            # Make sure audio exists
            if id1 not in audio_dict or id2 not in audio_dict:
                continue

            audio1 = audio_dict[id1]
            audio2 = audio_dict[id2]

            if audio1 is None or audio2 is None:
                continue

            # Encode lyrics
            lyrics1_enc = encode(lyrics1)
            lyrics2_enc = encode(lyrics2)
            
            # Pad lyrics to same length before subtraction
            max_len_lyrics = max(lyrics1_enc.shape[0], lyrics2_enc.shape[0])
            lyrics1_enc = nn.functional.pad(lyrics1_enc, (0, max_len_lyrics - lyrics1_enc.shape[0]))
            lyrics2_enc = nn.functional.pad(lyrics2_enc, (0, max_len_lyrics - lyrics2_enc.shape[0]))
            #lyrics_feat = torch.abs(lyrics1_enc.float() - lyrics2_enc.float())
            
            # Prepare audio features
            audio1_tensor = torch.tensor(audio1, dtype=torch.float32)
            audio2_tensor = torch.tensor(audio2, dtype=torch.float32)
            max_len_audio = max(audio1_tensor.shape[0], audio2_tensor.shape[0])
            audio1_tensor = nn.functional.pad(audio1_tensor, (0, max_len_audio - audio1_tensor.shape[0]))
            audio2_tensor = nn.functional.pad(audio2_tensor, (0, max_len_audio - audio2_tensor.shape[0]))
            #audio_feat = torch.abs(audio1_tensor - audio2_tensor)
            
            # Pad to equal final length for stacking
            # final_len = max(lyrics_feat.shape[0], audio_feat.shape[0])
            # print(final_len)
            # print(min(lyrics_feat.shape[0], audio_feat.shape[0]))
            # lyrics_feat = nn.functional.pad(lyrics_feat, (0, final_len - lyrics_feat.shape[0]))
            # audio_feat = nn.functional.pad(audio_feat, (0, final_len - audio_feat.shape[0]))
            
            # Stack as 2 channels
            lyrics_feat = torch.stack([lyrics1_enc, lyrics2_enc], dim=0)  # shape [2, final_len]
            audio_feat = torch.stack([audio1_tensor, audio2_tensor], dim=0)
            
            self.samples.append((lyrics_feat,audio_feat, torch.tensor(score1, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(batch):
        lyrics_feats,audio_feats, scores = zip(*batch)  # unpack
        max_len = max(feat.shape[1] for feat in lyrics_feats) 
        # find max length in this batch
    
        padded_lyrics_feats = []
        for feat in lyrics_feats:
            pad_width = max_len - feat.shape[1]
            if pad_width > 0:
                feat = nn.functional.pad(feat, (0, pad_width))  # pad only last dimension
            padded_lyrics_feats.append(feat)

        max_len = max(feat.shape[1] for feat in audio_feats) 
        # find max length in this batch
    
        padded_audio_feats = []
        for feat in audio_feats:
            pad_width = max_len - feat.shape[1]
            if pad_width > 0:
                feat = nn.functional.pad(feat, (0, pad_width))  # pad only last dimension
            padded_audio_feats.append(feat)
    
        audio_feats_tensor = torch.stack(padded_audio_feats)
        lyrics_feats_tensor = torch.stack(padded_lyrics_feats) # now safe to stack
        scores_tensor = torch.stack(scores)
    
        return audio_feats_tensor,lyrics_feats_tensor, scores_tensor
    
# --- 5. Build Dataloaders ---
full_dataset = LyricsAudioDataset(lyrics_df, audio_dict)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LyricsAudioDataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=LyricsAudioDataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=LyricsAudioDataset.collate_fn)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# --- 6. Model Definition ---

class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # (batch, seq, channel) for LSTM

        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)  # (batch, hidden)

        x = torch.relu(self.fc1(h_n))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # output between 0 and 1
        return x.squeeze(1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Lyrics Encoder: Embeds text and uses BiLSTM + projection to match audio embedding_dim
class Lyrics_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Lyrics_Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.lyric_proj = nn.Linear(hidden_dim * 2, embedding_dim)  # project BiLSTM output to embedding_dim

    def forward(self, input_seq):
        x = self.embedding(input_seq)         # (B, seq_len, embedding_dim)
        x, _ = self.lstm1(x)                  # (B, seq_len, hidden_dim * 2)
        x, _ = self.lstm2(x)                  # (B, seq_len, hidden_dim * 2)
        x = torch.mean(x, dim=1)              # mean pooling across time → (B, hidden_dim * 2)
        x = self.lyric_proj(x)                # project to (B, embedding_dim)
        return F.normalize(x, p=2, dim=1)     # L2 normalize



class AudioEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(AudioEmbeddingCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),  # [B, 16, ~110250]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),  # [B, 16, ~27562]

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),  # [B, 32, ~13781]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # [B, 32, ~3445]

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),  # [B, 64, ~1723]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # [B, 64, ~430]

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),  # [B, 128, ~215]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # [B, 128, 1]
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        """
        x: [B, 1, L]  - raw audio waveform
        """
        x = self.conv_block(x)  # [B, 128, 1]
        x = x.squeeze(-1)       # [B, 128]
        x = self.fc(x)          # [B, embedding_dim]
        return F.normalize(x, p=2, dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(SiameseNetwork, self).__init__()
        self.lyrics_encoder = Lyrics_Encoder(vocab_size, embedding_dim, hidden_dim)
        self.audio_encoder = AudioEmbeddingCNN(embedding_dim)

    def forward(self, x1_audio, x2_audio, x1_lyrics, x2_lyrics, merge_type='concat'):
        audio1 = self.audio_encoder(x1_audio)
        audio2 = self.audio_encoder(x2_audio)
        lyrics1 = self.lyrics_encoder(x1_lyrics)
        lyrics2 = self.lyrics_encoder(x2_lyrics)

        if merge_type == 'concat':
            merged = torch.cat((audio1, audio2, lyrics1, lyrics2), dim=-1)
        elif merge_type == 'subtract':
            merged = audio1 - audio2
        elif merge_type == 'multiply':
            merged = audio1 * audio2
        elif merge_type == 'mean':
            merged = (audio1 + audio2) / 2
        elif merge_type == 'cosine_similarity':
            merged = F.cosine_similarity(audio1, audio2).unsqueeze(1)
        else:
            raise ValueError("Unsupported merge type.")

        return merged

# Decoder: Predicts similarity from merged features
class SimilarityDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(SimilarityDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, merged_features):
        x = F.normalize(merged_features, p=2, dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Full Model: Combines encoder + similarity prediction
class FullModel(nn.Module):
    def __init__(self,  embedding_dim=128, hidden_dim=256):
        super(FullModel, self).__init__()
        self.siamese_network = SiameseNetwork(vocab_size, embedding_dim, hidden_dim)
        self.similarity_decoder = SimilarityDecoder(embedding_dim * 4)  # concat of 4 embeddings

    def forward(self, x1, x2, merge_type='concat'):
        x1_audio, x2_audio = x1[:, 0, :], x1[:, 1, :]
        x1_lyrics , x2_lyrics=x2[:,0,:],x2[:,1,:]
        x1_audio = x1_audio.unsqueeze(1)  # Shape becomes [32, 1, 220500]
        x2_audio = x2_audio.unsqueeze(1)  # Shape becomes [32, 1, 220500]
        merged_features = self.siamese_network(x1_audio, x2_audio, x1_lyrics, x2_lyrics, merge_type)
        output = self.similarity_decoder(merged_features)
        return output


# --- 7. Training Loop ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train(model, loader):
    model.train()
    total_loss = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x1.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x1.size(0)
    return total_loss / len(loader.dataset)

# Track losses
train_losses = []
val_losses = []

n_epochs = 30
for epoch in range(n_epochs):
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


torch.save(model.state_dict(), "siamese_model_weights.pth")


# --- 8. Final Testing ---

test_loss = evaluate(model, test_loader)
print(f"Test Loss = {test_loss:.4f}")

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


model.eval()
true_scores = []
predicted_scores = []
with torch.no_grad():
    for x1, x2, labels in test_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
        outputs = model(x1, x2).squeeze()
        predicted_scores.extend(outputs.cpu().numpy())
        true_scores.extend(labels.cpu().numpy())
errors = [abs(t - p) for t, p in zip(true_scores, predicted_scores)]
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.show()
