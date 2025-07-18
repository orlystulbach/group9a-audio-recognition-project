import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers

# Path to the dataset
DATA_DIR = 'smaller_dataset'
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32
MODEL_PATH = 'digit_classifier_model.h5'

# 1. Load data and extract features
def extract_features(file_path, n_mfcc=N_MFCC):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

X, y = [], []
for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if fname.endswith('.wav'):
            # Label is the last number before .wav, e.g., 0 in 0_02_0.wav
            label = fname.split('_')[-1].split('.')[0]
            file_path = os.path.join(folder_path, fname)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# 2. Encode labels (one-hot)
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
if y_encoded.shape[1] == 1:
    # If only two classes, LabelBinarizer returns shape (n, 1), so fix for softmax
    y_encoded = np.hstack([1 - y_encoded, y_encoded])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# 4. Build the model
model = keras.Sequential([
    layers.Input(shape=(N_MFCC,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {acc * 100:.2f}%')

# 7. Save the model
model.save(MODEL_PATH)
print(f'Model saved to {MODEL_PATH}') 


