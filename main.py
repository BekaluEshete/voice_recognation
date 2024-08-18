

import os
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Constants
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
OUTPUT_FILENAME = "output.wav"
DATASET_PATH = 'archive'  # Path to your RAVDESS dataset folder
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  # RAVDESS emotions

def extract_features_from_file(file_path):
    signal, rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def load_dataset(dataset_path):
    features = []
    labels = []
    for actor_folder in os.listdir(dataset_path):
        actor_folder_path = os.path.join(dataset_path, actor_folder)
        if os.path.isdir(actor_folder_path):
            for file_name in os.listdir(actor_folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(actor_folder_path, file_name)
                    emotion_label = int(file_name.split('-')[2]) - 1  # Adjust index for emotion label
                    emotion = EMOTIONS[emotion_label]
                    feature = extract_features_from_file(file_path)
                    features.append(feature)
                    labels.append(emotion)
    return np.array(features), np.array(labels)

def train_model():
    # Load and preprocess the dataset
    X, y = load_dataset(DATASET_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train an SVM model
    model = SVC()
    model.fit(X_scaled, y)
    
    # Save the model and scaler to files
    joblib.dump(model, 'emotion_recognition_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model and scaler trained and saved successfully.")

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER)
    
    print("Start Recording for 5 seconds ...")
    frames = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(0, FRAMES_PER_BUFFER)
    line, = ax.plot(x, np.random.rand(FRAMES_PER_BUFFER))
    ax.set_ylim([-32768, 32767])
    ax.set_xlim([0, FRAMES_PER_BUFFER])
    plt.show()

    for i in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        signal = np.frombuffer(data, dtype=np.int16)
        line.set_ydata(signal)
        fig.canvas.draw()
        fig.canvas.flush_events()

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")
    return frames

def save_audio(frames, filename):
    p = pyaudio.PyAudio()
    with wave.open(filename, "wb") as obj:
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b"".join(frames))
    print(f"Audio saved as {filename}")

def plot_saved_audio(filename):
    try:
        with wave.open(filename, "rb") as obj:
            n_samples = obj.getnframes()
            sample_freq = obj.getframerate()
            n_channels = obj.getnchannels()
            signal_wave = obj.readframes(-1)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    
    t_audio = n_samples / sample_freq
    print(f"Audio duration: {t_audio:.2f} seconds")
    
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    
    if n_channels == 2:
        signal_array = signal_array.reshape(-1, 2)
        signal_array = signal_array[:, 0]
    
    times = np.linspace(0, t_audio, num=n_samples)
    
    plt.ioff()
    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_array[:n_samples])
    plt.title("Audio Signal")
    plt.ylabel("Signal wave")
    plt.xlabel("Time (s)")
    plt.xlim(0, t_audio)
    plt.show()

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription: " + text)
            with open("transcription.txt", "w") as file:
                file.write(text)
            print("Transcription saved vs as transcription.txt")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

def extract_features(filename):
    signal, rate = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_emotion(features, model, scaler):
    features = scaler.transform([features])
    emotion = model.predict(features)
    return emotion[0]

# Main execution
if __name__ == "__main__":
    # Train the model with the RAVDESS dataset
    train_model()
    
    frames = record_audio()
    save_audio(frames, OUTPUT_FILENAME)
    plot_saved_audio(OUTPUT_FILENAME)
    transcribe_audio(OUTPUT_FILENAME)

    # Load the pre-trained model and scaler
    model = joblib.load('emotion_recognition_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Extract features from the recorded audio
    features = extract_features(OUTPUT_FILENAME)

    # Predict the emotion
    predicted_emotion = predict_emotion(features, model, scaler)
    print(f"Predicted Emotion: {predicted_emotion}")
