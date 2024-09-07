import os
import time
import wave
import numpy as np
import deepspeech

# Define the paths to the model files and audio directory
model_path = "C:/Users/Lenovo/OneDrive - Alexandria University/Desktop/Projects/AudioTranscription/DeepSpeechModel/deepspeech-0.9.3-models.pbmm"
scorer_path = "C:/Users/Lenovo/OneDrive - Alexandria University/Desktop/Projects/AudioTranscription/DeepSpeechModel/deepspeech-0.9.3-models.scorer"
audio_dir = "C:/Users/Lenovo/OneDrive - Alexandria University/Desktop/Projects/AudioTranscription/Audio_Files"
output_dir = "C:/Users/Lenovo/OneDrive - Alexandria University/Desktop/Projects/AudioTranscription/DeepSpeechModel/DeepFakeTranscription"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the DeepSpeech model
model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)


# Function to transcribe a single audio file
def transcribe_audio(file_path):
    with wave.open(file_path, 'rb') as audio_wave:
        frames = audio_wave.readframes(audio_wave.getnframes())
        audio_np = np.frombuffer(frames, np.int16)
        duration = audio_wave.getnframes() / audio_wave.getframerate()
    transcription = model.stt(audio_np)
    return transcription, duration


# Process each audio file and record transcription time and duration
times = []
durations = []
for i in range(1, 13):  # Assuming files are named from 1.wav to 11.wav
    audio_file = os.path.join(audio_dir, f"{i}.wav")
    output_file = os.path.join(output_dir, f"transcription{i}_deepspeech.txt")

    start_time = time.time()
    transcription, audio_duration = transcribe_audio(audio_file)
    end_time = time.time()

    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    durations.append(audio_duration)

    with open(output_file, 'w') as f:
        f.write(transcription)

# Calculate weighted average transcription time
total_time = sum(times)
total_duration = sum(durations)
weighted_average_time = total_time / total_duration

# Create a summary of the results
summary_path = os.path.join(output_dir, "transcription_summary.txt")
with open(summary_path, 'w') as f:
    f.write("Transcription Times and Audio Durations:\n")
    for idx, (t, d) in enumerate(zip(times, durations), 1):
        f.write(f"File {idx}: Time = {t:.2f} seconds, Duration = {d:.2f} seconds\n")
    f.write(f"\nWeighted Average Transcription Time: {weighted_average_time:.2f} seconds per second of audio\n")

# Output the paths to the results for easy access
output_dir, summary_path