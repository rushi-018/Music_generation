# Count the frequency of each tag
import pandas as pd
import re

import os
import shutil
from pathlib import Path

annotations_path = 'data/annotations.csv'
clip_info_path = 'data/clip_info.csv'

if not os.path.exists(annotations_path):
	raise FileNotFoundError(f"{annotations_path} does not exist.")
if not os.path.exists(clip_info_path):
	raise FileNotFoundError(f"{clip_info_path} does not exist.")

annotations = pd.read_csv(annotations_path, sep='\t')
clip_info = pd.read_csv(clip_info_path, sep='\t')

# Ensure 'clip_id' column exists in annotations
if 'clip_id' not in annotations.columns:
    raise KeyError("'clip_id' column is missing from annotations.")

tag_counts = annotations.iloc[:, 1:].select_dtypes(include='number').sum().sort_values(ascending=False)
# print(tag_counts)
tag_to_scenario = {
    'calm': 'relaxation',
    'relaxing': 'relaxation',
    'energetic': 'focus',
    'soft': 'sleep',
    # Add more mappings as needed
    'happy': 'joy',
    'sad': 'melancholy',
    'angry': 'aggression',
    'peaceful': 'relaxation',
    'upbeat': 'energy',
    'romantic': 'love',
    'melancholic': 'melancholy',
    'excited': 'energy',
    'calming': 'relaxation',
    'intense': 'focus',
    'dark': 'melancholy',
    'bright': 'joy',
    'mellow': 'relaxation',
    'soothing': 'sleep',
    'uplifting': 'joy',
    'chill': 'relaxation',
    'motivational': 'focus',
    'groovy': 'energy',
    'funky': 'energy',
    'dreamy': 'sleep',
    'sentimental': 'love',
    'nostalgic': 'melancholy',
    'spiritual': 'relaxation',
    'epic': 'focus',
    'ambient': 'relaxation',
    'happy': 'joy',
    'sad': 'melancholy',
    'angry': 'aggression',
    'peaceful': 'relaxation',
    'upbeat': 'energy',
    'romantic': 'love',
    'melancholic': 'melancholy',
    'excited': 'energy',
    'calming': 'relaxation',
    'intense': 'focus',
    'dark': 'melancholy',
    'bright': 'joy',
    'mellow': 'relaxation',
    'soothing': 'sleep',
    'uplifting': 'joy',
    'chill': 'relaxation',
    'motivational': 'focus',
    'groovy': 'energy',
    'funky': 'energy',
    'dreamy': 'sleep',
    'sentimental': 'love',
    'nostalgic': 'melancholy',
    'spiritual': 'relaxation',
    'epic': 'focus',
    'ambient': 'relaxation'
}

# Add a 'scenario' column to the annotations
annotations['scenario'] = annotations.apply(
    lambda row: next((scenario for tag, scenario in tag_to_scenario.items() if tag in row.index and row[tag] == 1), None), axis=1
)
# print(annotations[['clip_id', 'scenario']].head())
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import soundfile as sf
import json

# Update paths
DATA_DIR = 'data/mp3'  # Parent directory containing MP3 folders
ANNOTATIONS_PATH = 'data/annotations.csv'
CLIP_INFO_PATH = 'data/clip_info.csv'
OUTPUT_DIR = 'generated_music'

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # seconds

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion/scenario mapping
EMOTION_MAPPING = {
    'relaxation': 0,
    'focus': 1,
    'sleep': 2,
    'joy': 3,
    'melancholy': 4,
    'aggression': 5,
    'energy': 6,
    'love': 7
}

def load_or_create_file_mapping(annotations_df, clip_info_df, mapping_cache_file='data/file_mapping.json'):
    """Load mapping from cache or create new one"""
    if os.path.exists(mapping_cache_file):
        print("Loading file mapping from cache...")
        with open(mapping_cache_file, 'r') as f:
            return json.load(f)
    
    print("Creating new file mapping...")
    file_mapping = {}
    
    # Merge annotations with clip_info to get mp3_path
    merged_df = pd.merge(annotations_df, clip_info_df, on='clip_id', how='inner')
    
    for _, row in merged_df.iterrows():
        clip_id = str(row['clip_id'])
        mp3_path = row['mp3_path']
        
        if pd.notna(mp3_path) and mp3_path.strip():  # Check if mp3_path is valid
            # Check in both possible locations
            potential_paths = [
                os.path.join('data', 'mp3', 'zip1_files', mp3_path),
                os.path.join('data', 'mp3', 'zip2_files', mp3_path),
                os.path.join('data', 'mp3', 'zip3_files', mp3_path)
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    file_mapping[clip_id] = path
                    break
    
    print(f"Found {len(file_mapping)} valid file mappings")
    
    # Save mapping to cache
    os.makedirs(os.path.dirname(mapping_cache_file), exist_ok=True)
    with open(mapping_cache_file, 'w') as f:
        json.dump(file_mapping, f)
    
    return file_mapping

def extract_features(file_path):
    """Extract audio features from a file"""
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, duration=29.0, sr=22050)  # Fixed duration and sample rate
        
        if len(y) == 0:
            print(f"Empty audio file: {file_path}")
            return None
            
        # Ensure minimum length
        if len(y) < sr:  # Less than 1 second
            print(f"Audio file too short: {file_path}")
            return None
        
        # Get fixed-length features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Calculate statistics for each feature
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def prepare_dataset():
    """Prepare dataset from annotations"""
    try:
        # Load data
        annotations = pd.read_csv('data/annotations.csv', sep='\t')
        clip_info = pd.read_csv('data/clip_info.csv', sep='\t')
        
        # Get file mapping
        file_mapping = load_or_create_file_mapping(annotations, clip_info)
        
        if not file_mapping:
            raise Exception("No valid file mappings found")
        
        # Extract features
        features_list = []
        valid_files = []
        
        print("\nExtracting features...")
        for clip_id, file_path in file_mapping.items():
            if os.path.exists(file_path):
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    valid_files.append(file_path)
                    print(f"Successfully processed: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        if not features_list:
            raise Exception("No valid features extracted")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        print(f"\nFeature extraction complete. Shape: {X.shape}")
        
        return X, valid_files
        
    except Exception as e:
        print(f"\nError in dataset preparation: {str(e)}")
        return None, None

def create_model(input_shape):
    """Create the music generation model"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_shape, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_music(model, input_features, output_path, duration=30, sr=22050):
    """Generate music from features and save to file."""
    print(f"\nGenerating music: {output_path}")
    
    # Generate new features
    generated_features = model.predict(input_features.reshape(1, -1))[0]
    
    # Convert features back to audio
    t = np.linspace(0, duration, sr * duration)
    audio = np.zeros_like(t)
    
    # Enhanced synthesis with harmonics
    base_freq = 220  # A3 note
    harmonics = 8
    
    for i in range(harmonics):
        frequency = base_freq * (1.05946 ** i)
        amplitude = generated_features[i % len(generated_features)]
        phase = 2 * np.pi * np.random.random()
        audio += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Save to file
    sf.write(output_path, audio, sr)
    print(f"Music saved to: {output_path}")

def main():
    """Main function to run the script"""
    try:
        print("\nStarting dataset preparation...")
        X, valid_files = prepare_dataset()
        
        if X is not None and valid_files:
            print("\nDataset preparation successful!")
            print(f"Features shape: {X.shape}")
            print(f"Number of valid files: {len(valid_files)}")
            
            # Create and train the model
            print("\nTraining model...")
            model = create_model(X.shape[1])
            model.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)
            
            # Generate music
            output_dir = 'generated_music'
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate 5 different music samples
            for i in range(5):
                output_path = os.path.join(output_dir, f'generated_music_{i+1}.wav')
                generate_music(model, X[i], output_path)
            
        else:
            print("\nDataset preparation failed!")
            
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        return

if __name__ == "__main__":
    main()
