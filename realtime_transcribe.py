import pyaudio
import numpy as np
import torch
import whisper
import threading
import queue
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Silence detection settings
SILENCE_THRESHOLD = 500  # Audio level threshold to be considered silence
SILENT_CHUNKS = 2 * (RATE // CHUNK)  # Number of silent chunks before processing (2 seconds)

# Whisper model configuration
MODEL_SIZE = "base.en"

# --- Translation Model Configuration (Gemma) ---
# Define the Hugging Face model name
TRANSLATION_MODEL_NAME = "google/gemma-2-2b-jpn-it"
# Define the local path to check for the model first
LOCAL_MODEL_PATH = os.path.join("models", "gemma-2-2b-jpn-it")

# --- Determine the model source ---
# Prioritize loading from the local path if it exists.
if os.path.isdir(LOCAL_MODEL_PATH):
    print(f"✅ Found local Gemma model at: {LOCAL_MODEL_PATH}")
    model_source = LOCAL_MODEL_PATH
else:
    print(f"ℹ️ Local Gemma model not found at '{LOCAL_MODEL_PATH}'.")
    print(f"   Will attempt to download from Hugging Face Hub: '{TRANSLATION_MODEL_NAME}'.")
    print("   Note: This requires an internet connection and may require authentication.")
    model_source = TRANSLATION_MODEL_NAME

# --- Global Shared Resources ---
audio_queue = queue.Queue()
text_to_translate_queue = queue.Queue()
transcribed_text = ""
translated_text = ""
text_lock = threading.Lock()
is_running = True


def capture_audio():
    """
    Captures audio from the microphone and puts it into a queue.
    """
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("🎤 Audio capture started. Speak English (Press Ctrl+C to stop).")
        while is_running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_queue.put(data)
            except Exception as e:
                print(f"Error in audio capture loop: {e}")
                break
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("🎤 Audio capture stopped.")


def transcribe_audio():
    """
    Transcribes audio from the queue using the Whisper model.
    """
    global transcribed_text

    print("🤖 Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_SIZE, device=device)
    print(f"🤖 Whisper model '{MODEL_SIZE}' loaded on {device}.")

    audio_buffer = bytearray()
    silent_chunks_count = 0

    while is_running:
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_buffer.extend(audio_chunk)
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

            # Check for silence
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                silent_chunks_count += 1
            else:
                silent_chunks_count = 0

            # Process audio on silence or if the buffer is too long
            if silent_chunks_count > SILENT_CHUNKS or len(audio_buffer) > RATE * 15:
                if len(audio_buffer) > RATE:  # Process only if there's enough audio
                    print("🤖 Detected pause, transcribing audio...")
                    audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    result = model.transcribe(audio_data, fp16=torch.cuda.is_available())
                    new_text = result.get('text', '').strip()

                    if new_text:
                        with text_lock:
                            transcribed_text = new_text
                        text_to_translate_queue.put(new_text)

                # Reset buffer and silence counter
                audio_buffer = bytearray()
                silent_chunks_count = 0
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during transcription: {e}")
            break


def translate_text_gemma():
    """
    Translates text from the queue using the Gemma model.
    """
    global translated_text

    print(f"🌐 Loading Gemma translation model from '{model_source}'...")
    print("   (This may take a while and require significant memory)")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        print("🌐 Gemma model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Gemma model: {e}")
        print(
            "   If downloading, please ensure you are logged in via 'huggingface-cli login' and have accepted the model's terms.")
        return

    while is_running:
        try:
            text_to_process = text_to_translate_queue.get(timeout=1)

            prompt = f"以下の英語の文章を日本語に翻訳してください。\nEnglish: \"{text_to_process}\""
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
            )

            response = tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            new_translated_text = response.strip()

            with text_lock:
                translated_text = new_translated_text

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during translation: {e}")
            break


def main():
    """
    Main function to start threads and display results in the console.
    """
    global is_running

    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
    translate_thread = threading.Thread(target=translate_text_gemma, daemon=True)

    audio_thread.start()
    transcribe_thread.start()
    translate_thread.start()

    last_displayed_transcribed = ""
    last_displayed_translated = ""

    try:
        while is_running:
            with text_lock:
                current_transcribed = transcribed_text
                current_translated = translated_text

            # Print new transcriptions and translations only if they have changed
            if current_transcribed and current_transcribed != last_displayed_transcribed:
                print("\n" + "="*50)
                print(f"🔴 TRANSCRIBED (EN): {current_transcribed}")
                last_displayed_transcribed = current_transcribed

            if current_translated and current_translated != last_displayed_translated:
                print(f"🟢 TRANSLATED  (JA): {current_translated}")
                print("="*50 + "\n")
                last_displayed_translated = current_translated

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        is_running = False

    # Wait for threads to finish
    audio_thread.join(timeout=2)
    transcribe_thread.join(timeout=2)
    translate_thread.join(timeout=2)
    print("✅ Shutdown complete.")


if __name__ == '__main__':
    main()