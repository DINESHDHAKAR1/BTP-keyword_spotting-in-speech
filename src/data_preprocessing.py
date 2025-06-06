import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import logging
import os

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, label_to_idx, sr=16000, n_mels=64, n_fft=1024, hop_length=512, spectrogram_shape=(64, 32), log_dir="logs"):
        self.file_paths = file_paths
        self.labels = [label_to_idx[label] for label in labels]
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram_shape = spectrogram_shape
        self.logger = logging.getLogger("AudioDataset")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "data_preprocessing.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            y, sr = librosa.load(self.file_paths[idx], sr=self.sr)
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                         n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            # Ensure consistent shape (pad or truncate)
            if spectrogram_db.shape[1] > self.spectrogram_shape[1]:
                spectrogram_db = spectrogram_db[:, :self.spectrogram_shape[1]]
            elif spectrogram_db.shape[1] < self.spectrogram_shape[1]:
                pad_width = self.spectrogram_shape[1] - spectrogram_db.shape[1]
                spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
            self.logger.debug(f"Processed audio file: {self.file_paths[idx]}")
            return torch.tensor(spectrogram_db, dtype=torch.float32).unsqueeze(0), self.labels[idx]  # Add channel dim

        except Exception as e:
            self.logger.error(f"Failed to process audio file {self.file_paths[idx]}: {str(e)}")
            raise