import cv2
import pyaudio
import numpy as np
import torch
import whisper
import threading
import queue
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Silence detection settings
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * (RATE // CHUNK)

# Whisper model configuration
MODEL_SIZE = "base.en"

# --- Translation Model Configuration (Gemma) ---
# Define the Hugging Face model name
TRANSLATION_MODEL_NAME = "google/gemma-2-2b-jpn-it"
# Define the local path to check for the model first
LOCAL_MODEL_PATH = os.path.join("models", "gemma-2-2b-jpn-it")

# --- Font Configuration for UI ---
JAPANESE_FONT_PATH = "fonts/Noto_Sans_JP/static/NotoSansJP-Regular.ttf"

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
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("🎤 Audio capture started. Speak English.")
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

            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                silent_chunks_count += 1
            else:
                silent_chunks_count = 0

            if silent_chunks_count > SILENT_CHUNKS or len(audio_buffer) > RATE * 15:
                if len(audio_buffer) > RATE:
                    print("🤖 Detected pause, transcribing audio...")
                    audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    result = model.transcribe(audio_data, fp16=torch.cuda.is_available())
                    new_text = result.get('text', '').strip()

                    if new_text:
                        print(f"   -> Transcribed: {new_text}")
                        with text_lock:
                            transcribed_text = new_text
                        text_to_translate_queue.put(new_text)

                audio_buffer = bytearray()
                silent_chunks_count = 0
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during transcription: {e}")
            break


def translate_text_gemma():
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
        print(f"🌐 Gemma model loaded successfully.")
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

            print(f"   -> Translated (ja): {new_translated_text}")
            with text_lock:
                translated_text = new_translated_text

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error during translation: {e}")
            break


def main():
    global is_running

    try:
        jp_font = ImageFont.truetype(JAPANESE_FONT_PATH, 32)
        print(f"✅ Successfully loaded Japanese font: {JAPANESE_FONT_PATH}")
    except IOError:
        print(f"❌ Error: The font file '{JAPANESE_FONT_PATH}' was not found.")
        jp_font = ImageFont.load_default()

    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
    translate_thread = threading.Thread(target=translate_text_gemma, daemon=True)

    audio_thread.start()
    transcribe_thread.start()
    translate_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("🎥 Video capture started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with text_lock:
            display_text_orig = f"EN: {transcribed_text}"
            display_text_trans = f"JA: {translated_text}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, display_text_orig, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((20, 70), display_text_trans, font=jp_font, fill=(100, 255, 100, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('Real-Time Speech Translation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Shutting down...")
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    audio_thread.join(timeout=2)
    transcribe_thread.join(timeout=2)
    translate_thread.join(timeout=2)
    print("✅ Shutdown complete.")


if __name__ == '__main__':
    main()
