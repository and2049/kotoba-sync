import cv2
import pyaudio
import numpy as np
import torch
import whisper
import threading
import queue
import time

# --- Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * (RATE // CHUNK)  # 2 seconds of silence
MODEL_SIZE = "base.en"

# --- Global Shared Resources ---
audio_queue = queue.Queue()
transcribed_text = ""
text_lock = threading.Lock()
# Use a global flag to signal threads to stop
is_running = True


def capture_audio():
    # Captures audio and puts it into a queue. Stops when is_running is False.
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎤 Audio capture started. Speak into the microphone.")

    while is_running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            print(f"Error in audio capture: {e}")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("🎤 Audio capture stopped.")


def transcribe_audio():
    # Processes audio from the queue using Whisper. The transcription happens after a period of silence.
    global transcribed_text, is_running

    print("🤖 Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_SIZE, device=device)
    print(f"🤖 Whisper model '{MODEL_SIZE}' loaded on {device}.")

    audio_buffer = bytearray()
    silent_chunks_count = 0

    while is_running:
        try:
            # Wait for audio data. The timeout allows the loop to check is_running.
            audio_chunk = audio_queue.get(timeout=1)

            audio_buffer.extend(audio_chunk)
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                silent_chunks_count += 1
            else:
                silent_chunks_count = 0

            # Only transcribe if we have a significant pause (silence)
            if silent_chunks_count > SILENT_CHUNKS:
                if len(audio_buffer) > RATE:  # Process if there's more than 1s of audio
                    print("🤫 Silence detected, processing audio...")

                    # Convert byte buffer to float32 numpy array
                    audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    # Transcribe
                    result = model.transcribe(audio_data, fp16=torch.cuda.is_available())
                    new_text = result['text'].strip()

                    if new_text:
                        # Print transcription to terminal
                        print(f"   [Terminal Output] -> {new_text}")
                        with text_lock:
                            transcribed_text = new_text

                # Clear buffer and silence count after processing
                audio_buffer = bytearray()
                silent_chunks_count = 0

        except queue.Empty:
            # This is expected when no one is speaking.
            continue
        except Exception as e:
            print(f"Error during transcription: {e}")
            break


def main():
    # Main function to run video capture and display.
    global transcribed_text, is_running

    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)

    audio_thread.start()
    transcribe_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("🎥 Video capture started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with text_lock:
            display_text = transcribed_text

        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Transcription', frame)

        if cv2.waitKey(1) == ord('q'):
            print("🛑 Shutting down...")
            is_running = False  # Signal threads to stop
            break

    # Wait for threads to finish their work
    audio_thread.join()
    transcribe_thread.join()

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete.")


if __name__ == '__main__':
    main()