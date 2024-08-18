      


Bahir Dar Institute of Technology Bahir Dar University Faculty of Computing Departments of Software  Eng
Machine Learning

Project Documentation: Emotion Recognition Using RAVDESS Dataset


1. Introduction
This project involves building an emotion recognition system using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system will record audio, preprocess it, and predict the emotion expressed in the audio using a pre-trained Support Vector Machine (SVM) model.
2. Requirements
To run this project, the following Python packages are required:
numpy
pyaudio
matplotlib
speech_recognition
librosa
joblib
scikit-learn
You can install these packages using pip:

pip install numpy pyaudio matplotlib SpeechRecognition librosa joblib scikit-learn

3. Project Structure
The project directory should have the following structure:

project_folder/
│
├── archive/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── ...
│   └── Actor_24/
│
├── emotion_recognition_model.pkl
├── scaler.pkl
├── output.wav
└── main.py

archive/: Contains the RAVDESS dataset with subfolders for each actor.
emotion_recognition_model.pkl: Saved SVM model.
scaler.pkl: Saved scaler used for feature normalization.
output.wav: Recorded audio file.
main.py: Main script file.
4. Detailed Explanation
Data Preparation
The function load_dataset loads the RAVDESS dataset, extracts features using librosa, and labels the data based on the filenames.


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

Model Training
The function train_model loads the dataset, preprocesses the features using StandardScaler, trains an SVM model, and saves the trained model and scaler.

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

Recording and Saving Audio
The function record_audio uses pyaudio to record a 5-second audio sample and save it to a file.

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

Plotting Audio
The function plot_saved_audio plots the waveform of the saved audio file.

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

Transcribing Audio
The function transcribe_audio uses the speech_recognition library to transcribe the recorded audio to text using the Google Speech Recognition API.

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription: " + text)
            with open("transcription.txt", "w") as file:
                file.write(text)
            print("Transcription saved as transcription.txt")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

Feature Extraction
The function extract_features extracts MFCC (Mel-Frequency Cepstral Coefficients) features from the recorded audio file.

def extract_features(filename):
    signal, rate = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

Predicting Emotion
The function predict_emotion uses the pre-trained SVM model to predict the emotion from the extracted features.


def predict_emotion(features, model, scaler):
    features = scaler.transform([features])
    emotion = model.predict(features)
    return emotion[0]

5. Running the Project
To run the project, follow these steps:
Prepare the Dataset: Ensure the RAVDESS dataset is downloaded and organized in the archive folder as described.
Install Dependencies: Install the required Python packages using pip.
Run the Script: Execute the main script main.py to train the model, record audio, and predict the emotion.
6. Conclusion
This project demonstrates how to build an emotion recognition system using audio data. It covers data preprocessing, model training, audio recording, feature extraction, and emotion prediction. By leveraging the RAVDESS dataset and various Python libraries, the system can accurately predict emotions from speech.


