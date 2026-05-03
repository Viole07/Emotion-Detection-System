import os
import numpy as np
from speech_utils import extract_features
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# 1. Load Data
X, y = [], []
data_path = 'data/'
# RAVDESS emotion mapping
emotion_map = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fearful'}

print("Starting data loading... this may take a minute.")

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split('-')
            # RAVDESS filename format: 03-01-XX-... (3rd part is emotion)
            if len(parts) > 2:
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    try:
                        feature = extract_features(os.path.join(root, file))
                        X.append(feature)
                        y.append(int(emotion_code) - 1) # 0-indexed labels
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: No audio files found. Check if your 'data' folder contains Actor_XX folders.")
else:
    print(f"Successfully loaded {len(X)} audio files.")

    # 2. Build Simple MLP Model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(40,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='softmax') # 6 classes from RAVDESS
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training...")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    # 3. Save it
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/speech_emotion_model.h5')
    print("Speech model trained and saved in models/speech_emotion_model.h5!")