import os
import time
import whisper

# Define the paths
audio_dir = r"C:\Users\Lenovo\OneDrive - Alexandria University\Desktop\Projects\AudioTranscription\Audio_Files"
output_dir = r"C:\Users\Lenovo\OneDrive - Alexandria University\Desktop\Projects\AudioTranscription\Whisper\TranscriptionOutputs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the model
model = whisper.load_model("base")

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    print(f"Processing {file_path}...")
    start_time = time.time()
    result = model.transcribe(file_path)
    elapsed_time = time.time() - start_time
    return result['text'], elapsed_time

# Process each audio file and save the results
times = []
for i in range(1, 13):  # Process files from 1.wav to 12.wav
    audio_file = os.path.join(audio_dir, f"{i}.wav")
    output_file = os.path.join(output_dir, f"transcription{i}_whisper.txt")

    transcription, elapsed_time = transcribe_audio(audio_file)
    times.append(elapsed_time)

    # Save the transcription
    with open(output_file, 'w') as f:
        f.write(transcription)

    print(f"Transcription for {audio_file} completed in {elapsed_time:.2f} seconds.")

# Print a summary of times
average_time = sum(times) / len(times)
print("All transcriptions completed.")
print(f"Average processing time per file: {average_time:.2f} seconds")

# Optionally, you can write the times to a file as well
summary_path = os.path.join(output_dir, "transcription_summary.txt")
with open(summary_path, 'w') as f:
    f.write("Transcription Times for Each Audio File:\n")
    for index, time_taken in enumerate(times, 1):
        f.write(f"File {index}.wav: {time_taken:.2f} seconds\n")
    f.write(f"\nAverage Transcription Time: {average_time:.2f} seconds\n")