import librosa
import numpy as np

def extract_features(file_path):
    # Load audio file - Removed res_type to let librosa use 'soxr' automatically
    data, sample_rate = librosa.load(file_path)
    
    # Extract MFCCs (40 features)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs