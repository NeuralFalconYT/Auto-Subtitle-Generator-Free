# #@title whisper part
%cd /content/video_dub
from subtitle import whisper_api
import pysrt
import os
from utils import language_dict
source_lang_list = ['Automatic',"Bengali","Hindi","English"]

available_language=language_dict.keys()
source_lang_list.extend(available_language)  
target_language_list=source_lang_list
target_language_list.remove('Automatic')
def get_prompt(language):
    """
    Generates a dubbing-friendly translation prompt for an .srt subtitle file.
    Tailored for natural speech and timing accuracy.
    """
    prompt = f"""
-------------- You are a professional subtitle translator for **video dubbing**.
Translate the following `.srt` subtitle file into **{language}** while preserving timing, meaning, and emotional tone.

Output in JSON format exactly like this:

```json
{{
  "subtitle sequence number": {{
    "timestamp": "original timestamp",
    "actual subtitle text": "original English subtitle line",
    "translation": "natural, dubbing-friendly {language} translation"
  }}
}}
```

**Guidelines for Translation:**

1. **Understand the full context** before translating — read the entire subtitle file first.
2. Translate into **natural, conversational {language}**, not a direct word-for-word translation.
3. Make sure sentences flow **smoothly when spoken aloud** — prioritize oral rhythm over strict literal accuracy.
4. **Preserve tone, character personality, and emotional intent** (formal, casual, humorous, dramatic, etc.).
5. Adapt culturally — replace idioms, jokes, and references with ones that feel authentic to {language} audiences.
6. Keep translations **roughly similar in length** to the original so lip movements sync naturally.
7. Avoid overly complex or literary structures — prefer **clear, spoken language**.
8. If multiple sentences fit within the same timestamp, **keep them in the same style and pacing** as the original.
9. For Bengali, use Kolkata-style vocabulary, spelling, and pronunciation only — avoid Bangladeshi terms, spellings, and accent patterns. 
"""
    return prompt




def srt_to_txt(srt_path,target_language):
    txt_path = srt_path.replace(".srt", ".txt")
    subs = pysrt.open(srt_path, encoding='utf-8')

    with open(txt_path, 'w', encoding='utf-8') as f:
        for sub in subs:
            f.write(f"{sub.index}\n")
            f.write(f"{sub.start} --> {sub.end}\n")
            f.write(f"{sub.text}\n\n")
        f.write(get_prompt(target_language))
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return txt_path,content






def create_prompt(video, media_path=None, language=None,target_language="Bengali"):
    old_video=video
    try:
        srt_path = whisper_api(old_video, language)
        txt_path, prompt = srt_to_txt(srt_path,target_language)
        return os.path.abspath(srt_path), os.path.abspath(txt_path), prompt,os.path.abspath(old_video)
    except Exception as e:
        gr.Warning(f"Error during transcription or text extraction: {str(e)}")
        return None, None, None,old_video

##################################################srt start ######
#@title srt part
# pip install edge-tts pysrt librosa soundfile pydub tqdm

import os
import re
import uuid
import shutil
import platform
import datetime
import subprocess
import asyncio
import edge_tts
import pysrt
import librosa
import soundfile as sf
from tqdm.auto import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence

# (All configuration and utility functions from the previous answer remain the same)
# ...
# ---------------------- Voice Configuration ----------------------

# A selection of voices from edge-tts.
# You can get a full list by running: edge-tts --list-voices
from utils import female_voice_list,male_voice_list
# female_voice_list={
#  'Bengali': 'bn-BD-NabanitaNeural',
#  'English': "en-US-AvaMultilingualNeural", #'en-AU-NatashaNeural', #"en-IE-EmilyNeural"
#  'Hindi': 'hi-IN-SwaraNeural',
#  }
# male_voice_list= {
#  'Bengali': 'bn-BD-PradeepNeural',
#  'English': 'en-US-BrianMultilingualNeural', #"en-US-BrianNeural"
#  'Hindi': 'hi-IN-MadhurNeural',
#  }


# ---------------------- Async TTS Function ----------------------

async def text_to_speech_edge(text: str, voice: str, save_path: str):
    """
    Uses edge-tts to convert text to speech and saves it as an MP3 file.
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

# ---------------------- Utility Functions ----------------------

def get_subtitle_Dub_path(srt_file_path, language):
    """Constructs an output file path for the final dubbed audio."""
    file_name = os.path.splitext(os.path.basename(srt_file_path))[0]
    full_base_path = os.path.join(os.getcwd(), "TTS_DUB_RESULTS")
    os.makedirs(full_base_path, exist_ok=True)
    random_string = str(uuid.uuid4())[:6]
    new_path = os.path.join(full_base_path, f"{file_name}_{language}_{random_string}.wav")
    return new_path.replace("__", "_")
def clean_srt(input_path):
    """Removes noise characters like [♫] from subtitle text and saves a cleaned SRT."""
    def clean_srt_line(text):
        # Remove content within square brackets and specific characters
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        for bad_char in "♫♪":
            text = text.replace(bad_char, "")
        return text.strip()

    try:
        subs = pysrt.open(input_path, encoding='utf-8')
    except Exception:
        subs = pysrt.open(input_path, encoding='latin-1')  # Fallback

    # Prepare output path
    output_path = input_path.lower().replace(".srt", "") + "_.srt"
    output_dir = os.path.dirname(output_path)

    # ✅ Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write cleaned SRT
    with open(output_path, "w", encoding='utf-8') as file:
        for sub in subs:
            cleaned_text = clean_srt_line(sub.text)
            if cleaned_text:
                file.write(f"{sub.index}\n{sub.start} --> {sub.end}\n{cleaned_text}\n\n")

    return output_path

def prepare_srt(srt_path, target_language, translate=False):
    """Cleans an SRT file before dubbing."""
    if translate:
        print(f"Translating to {target_language} is not yet implemented.")
    path = clean_srt(srt_path)
    return path

def is_ffmpeg_installed():
    """Checks if FFmpeg is available on the system."""
    ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    try:
        subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True, ffmpeg_exe
    except Exception:
        print("WARNING: FFmpeg not found. Falling back to librosa for audio speedup, which may be slower.")
        return False, ffmpeg_exe

def atempo_chain(factor):
    """Creates an FFmpeg atempo filter chain for speed factors outside the 0.5-2.0 range."""
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor:.3f}"
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor *= 2.0
    parts.append(f"atempo={factor:.3f}")
    return ",".join(parts)

def speedup_audio_librosa(input_file, output_file, speedup_factor):
    """Changes audio speed using librosa (fallback)."""
    try:
        y, sr = librosa.load(input_file, sr=None)
        y_stretched = librosa.effects.time_stretch(y, rate=speedup_factor)
        sf.write(output_file, y_stretched, sr)
    except Exception as e:
        print(f"WARNING: Librosa speedup failed: {e}. Copying original audio.")
        shutil.copy(input_file, output_file)

def change_speed(input_file, output_file, speedup_factor, use_ffmpeg, ffmpeg_path):
    """Changes audio speed, preferring FFmpeg if available."""
    if use_ffmpeg:
        try:
            subprocess.run(
                [ffmpeg_path, "-i", input_file, "-filter:a", atempo_chain(speedup_factor), output_file, "-y"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"ERROR: FFmpeg speedup failed: {e}. Falling back to librosa.")
            speedup_audio_librosa(input_file, output_file, speedup_factor)
    else:
        speedup_audio_librosa(input_file, output_file, speedup_factor)

def remove_edge_silence(input_path, output_path):
    """Removes silence from the start and end of an audio file."""
    # y, sr = librosa.load(input_path, sr=None)
    # trimmed_audio, _ = librosa.effects.trim(y, top_db=25)
    # sf.write(output_path, trimmed_audio, sr)
    shutil.copy(input_path, output_path)

def remove_internal_silence(file_path, min_silence_len=100, silence_thresh=-45, keep_silence=50):
    """Removes long silences from within the audio file using pydub."""
    output_path = file_path.replace(".wav", "_no_silence.wav")
    sound = AudioSegment.from_file(file_path, format="wav")
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    combined = sum(audio_chunks, AudioSegment.empty())
    combined.export(output_path, format="wav")
    return output_path

def get_audio_duration_ms(file_path):
    """Returns the duration of an audio file in milliseconds."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        return int(librosa.get_duration(y=y, sr=sr) * 1000)
    except Exception:
        return 0

# ---------------------- Main Class ----------------------
class SRTDubbing:
    def __init__(self, use_ffmpeg=True, ffmpeg_path="ffmpeg"):
        self.use_ffmpeg = use_ffmpeg
        self.ffmpeg_path = ffmpeg_path
        self.cache_dir = "./cache"
        self.dummy_dir = "./dummy_srt_processing"
        os.makedirs(self.dummy_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    # ... (get_avg_speaker_speed, get_speed_factor, merge_fast_entries, etc. are unchanged)
    @staticmethod
    def get_avg_speaker_speed(srt_path, default_rate=14):
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(srt_path, encoding='latin-1')
        speeds = []
        for sub in subs:
            duration_sec = (sub.end.ordinal - sub.start.ordinal) / 1000
            char_count = len(sub.text.replace(" ", ""))
            if duration_sec > 0 and char_count > 0:
                speeds.append(char_count / duration_sec)
        return sum(speeds) / len(speeds) if speeds else default_rate

    @staticmethod
    def get_speed_factor(srt_path, default_tts_rate=14):
        avg_rate = SRTDubbing.get_avg_speaker_speed(srt_path)
        speed_factor = avg_rate / default_tts_rate if default_tts_rate > 0 else 1.0
        return round(speed_factor, 2)

    @staticmethod
    def merge_fast_entries(entries, max_pause_gap=800, max_merged_duration_ms=8000):
        if not entries: return []
        merged = []
        i = 0
        n = len(entries)
        while i < n:
            curr = entries[i].copy()
            j = i + 1
            while j < n:
                next_ = entries[j]
                gap = next_["start_time"] - curr["end_time"]
                new_duration = next_["end_time"] - curr["start_time"]
                if gap > max_pause_gap or new_duration > max_merged_duration_ms: break
                if not curr["text"].strip().endswith((".", "!", "?", ",")):
                    curr["text"] = curr["text"].strip() + ","
                curr["text"] += " " + next_["text"]
                curr["end_time"] = next_["end_time"]
                j += 1
            merged.append(curr)
            i = j
        return merged

    @staticmethod
    def convert_to_millisecond(t):
        return t.hours * 3600000 + t.minutes * 60000 + t.seconds * 1000 + t.milliseconds

    def read_srt_file(self, file_path):
        try:
            subs = pysrt.open(file_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(file_path, encoding='latin-1')
        entries = []
        prev_end = 0
        for idx, sub in enumerate(subs, 1):
            start = self.convert_to_millisecond(sub.start)
            end = self.convert_to_millisecond(sub.end)
            pause = start - prev_end if idx > 1 else start
            entries.append({
                'entry_number': idx, 'start_time': start, 'end_time': end,
                'text': sub.text.strip(), 'pause_time': pause,
            })
            prev_end = end
        return self.merge_fast_entries(entries)


    async def text_to_speech_srt(self, text, audio_path, voice, actual_duration_ms, default_speed_factor=1.0):
        """
        Generates TTS and meticulously adjusts it to fit the target duration using a cascading logic.
        """
        TOLERANCE_MS = 50
        temp_files = []

        try:
            # --- Step 1: Initial TTS Generation and Conversion to WAV ---
            uid = str(uuid.uuid4())
            temp_mp3 = os.path.join(self.cache_dir, f"{uid}.mp3")
            temp_files.append(temp_mp3)
            await text_to_speech_edge(text, voice, temp_mp3)

            path_to_process = os.path.join(self.cache_dir, f"{uid}_initial.wav")
            temp_files.append(path_to_process)
            AudioSegment.from_mp3(temp_mp3).export(path_to_process, format="wav")

            # --- Step 2: Apply Global Speaker Speed Factor by Default ---
            if default_speed_factor != 1.0:
                speed_adjusted_wav = os.path.join(self.cache_dir, f"{uid}_speed_adjusted.wav")
                temp_files.append(speed_adjusted_wav)
                change_speed(path_to_process, speed_adjusted_wav, default_speed_factor, self.use_ffmpeg, self.ffmpeg_path)
                path_to_process = speed_adjusted_wav

            # --- Step 3: Cascade of Corrections if Audio is Too Long ---
            current_duration_ms = get_audio_duration_ms(path_to_process)

            if current_duration_ms > actual_duration_ms + TOLERANCE_MS:
                # Action 1: Remove edge silence
                edge_trimmed_wav = os.path.join(self.cache_dir, f"{uid}_edge_trimmed.wav")
                temp_files.append(edge_trimmed_wav)
                remove_edge_silence(path_to_process, edge_trimmed_wav)
                path_to_process = edge_trimmed_wav
                current_duration_ms = get_audio_duration_ms(path_to_process)

                # Check again
                if current_duration_ms > actual_duration_ms + TOLERANCE_MS:
                    # Action 2: Remove internal silence
                    internal_trimmed_wav = remove_internal_silence(path_to_process) # This function creates its own output path
                    temp_files.append(internal_trimmed_wav)
                    path_to_process = internal_trimmed_wav
                    current_duration_ms = get_audio_duration_ms(path_to_process)

                    # Check again
                    if current_duration_ms > actual_duration_ms + TOLERANCE_MS:
                        # Action 3: Final speed adjustment (last resort)
                        final_factor = current_duration_ms / actual_duration_ms
                        change_speed(path_to_process, audio_path, final_factor, self.use_ffmpeg, self.ffmpeg_path)
                        return # Final version is at audio_path, so we can exit

            # --- Step 4: Finalization (Padding or Moving the file) ---
            final_duration_ms = get_audio_duration_ms(path_to_process)

            if abs(final_duration_ms - actual_duration_ms) <= TOLERANCE_MS:
                shutil.move(path_to_process, audio_path)
            elif final_duration_ms < actual_duration_ms:
                # Pad with silence at the end
                silence_to_add = actual_duration_ms - final_duration_ms
                final_audio = AudioSegment.from_file(path_to_process) + AudioSegment.silent(duration=silence_to_add)
                final_audio.export(audio_path, format="wav")
            else: # Should not happen often due to above logic, but as a fallback
                shutil.move(path_to_process, audio_path)

        finally:
            # Clean up all generated temporary files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

    @staticmethod
    def make_silence(duration, path):
        if duration > 0:
            AudioSegment.silent(duration=duration).export(path, format="wav")

    def create_folder_for_srt(self, srt_file_path):
        base = os.path.splitext(os.path.basename(srt_file_path))[0]
        folder = os.path.join(self.dummy_dir, f"{base}_{str(uuid.uuid4())[:4]}")
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def concatenate_audio_files(paths, output_path):
        if not paths:
            print("Warning: No audio segments to concatenate.")
            return

        # Filter out paths for zero-duration silence files which may not be created
        valid_paths = [p for p in paths if os.path.exists(p)]
        if not valid_paths:
            print("Warning: No valid audio segments found to concatenate.")
            return

        combined = AudioSegment.from_file(valid_paths[0])
        for p in valid_paths[1:]:
            combined += AudioSegment.from_file(p)

        combined.export(output_path, format="wav")

    async def srt_to_dub(self, srt_path, output_path, language, gender, speaker_talk_speed=True):
        if gender.lower() == "male":
            voice = male_voice_list.get(language)
        else:
            voice = female_voice_list.get(language)

        if not voice:
            raise ValueError(f"No voice found for language '{language}' and gender '{gender}'.")

        print(f"Using voice: {voice}")

        entries = self.read_srt_file(srt_path)
        folder = self.create_folder_for_srt(srt_path)
        all_audio_paths = []

        default_speed_factor = self.get_speed_factor(srt_path) if speaker_talk_speed else 1.0
        if speaker_talk_speed:
            print(f"Original speaker's average speed requires a TTS speed factor of: {default_speed_factor}")

        for i, entry in enumerate(tqdm(entries, desc="Dubbing SRT entries")):
            pause_path = os.path.join(folder, f"{i}_pause.wav")
            self.make_silence(entry['pause_time'], pause_path)
            all_audio_paths.append(pause_path)

            tts_path = os.path.join(folder, f"{i}_tts.wav")
            duration = entry['end_time'] - entry['start_time']
            await self.text_to_speech_srt(entry['text'], tts_path, voice, duration, default_speed_factor)
            all_audio_paths.append(tts_path)

        print("Concatenating all audio segments...")
        self.concatenate_audio_files(all_audio_paths, output_path)

        print(f"Cleaning up temporary folder: {folder}")
        shutil.rmtree(folder)

# ---------------------- Entrypoint Function ----------------------
async def srt_process(srt_path, language="English", gender="Male", translate=False, speaker_talk_speed=True):
    if not os.path.exists(srt_path) or not srt_path.lower().endswith(".srt"):
        raise FileNotFoundError(f"Please provide a valid .srt file path. Got: {srt_path}")

    print("Starting SRT dubbing process...")
    use_ffmpeg, ffmpeg_path = is_ffmpeg_installed()

    print("1. Preparing SRT file...")
    processed_srt = prepare_srt(srt_path, language, translate)

    output_path = get_subtitle_Dub_path(srt_path, language)
    print(f"2. Final audio will be saved to: {output_path}")

    dubber = SRTDubbing(use_ffmpeg, ffmpeg_path)
    await dubber.srt_to_dub(processed_srt, output_path, language, gender, speaker_talk_speed)

    if os.path.exists(processed_srt):
        os.remove(processed_srt)

    print("Dubbing process completed successfully!")
    return output_path

import os
import shutil
# from IPython.display import Audio, display

# Async dubbing function
async def run_dubbing(srt_path: str, language="Bengali", gender="Female", speaker_talk_speed=True):
    """
    Run the SRT dubbing process and return the final audio path.
    Displays the audio in Jupyter/Colab.
    """
    try:
        print("\n--- Running SRT Dubbing Process ---")

        final_audio_path = await srt_process(
            srt_path=srt_path,
            language=language,
            gender=gender,
            speaker_talk_speed=speaker_talk_speed
        )

        print(f"✅ Success! Dubbed audio saved at: {final_audio_path}")
        # display(Audio(final_audio_path))
        return final_audio_path

    except Exception as e:
        import traceback
        print(f"\n❌ Error during dubbing: {e}")
        traceback.print_exc()
        return None

    finally:
        # Clean up
        for folder in ["./cache", "./dummy_srt_processing"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Cleaned up {folder}")
# # Replace with your actual SRT path
# srt_file_path = "/content/Whisper-Turbo-Subtitle/translated_srt/dadaji.srt"

# # Run and get the dubbed audio path
# final_audio_path = await run_dubbing(srt_file_path, language="Bengali", gender="Female")

##################################################srt end ######

import subprocess
import torch

def replace_audio_with_ffmpeg(video_path, audio_path):

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs("./save_video/", exist_ok=True)
    output_path = f"./save_video/{base_name}_.mp4"

    gpu = torch.cuda.is_available()
    if gpu:
        print("✅ CUDA GPU is available. Running on GPU.")
    else:
        print("❌ No CUDA GPU found. Falling back to CPU.")

    video_codec = "h264_nvenc" if gpu else "libx264"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", video_codec,
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.abspath(output_path)


def remove_video_silences(video_path, language):
    # Run auto-editor to remove silences
    command = f"auto-editor '{video_path}' --margin 0.2sec"
    result = os.system(command)

    # Extract base filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs("./save_video/", exist_ok=True)

    if result == 0:
        altered_path = f"./save_video/{base_name}_ALTERED.mp4"
        save_path = f"./save_video/{base_name}{language}_dub.mp4"

        # If altered file exists, rename and return full absolute path
        if os.path.exists(altered_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(altered_path, save_path)
            return os.path.abspath(save_path)
        else:
            print(f"❌ Expected output not found: {altered_path}")
            return None
    else:
        print("❌ auto-editor command failed.")
        return None
import os
import json

def make_srt(llm_translate, media_file="output"):
    os.makedirs("./translated_srt/", exist_ok=True)

    # Handle both dict or JSON string input
    if isinstance(llm_translate, str):
        try:
            data = json.loads(llm_translate)
        except json.JSONDecodeError as e:
            print("❌ Invalid JSON format:", e)
            return None
    else:
        data = llm_translate

    tts_text = ""
    base = os.path.splitext(os.path.basename(media_file))[0]
    srt_path = f"./translated_srt/{base}.srt"

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, val in enumerate(data.values()):
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{val['timestamp']}\n")
            srt_file.write(f"{val['translation']}\n\n")
            tts_text += val['translation'] + " "

    return os.path.abspath(srt_path)
            
async def video_edit(video_file,llm_translate,language,gender,remove_silence_from_video=False):
  srt_path=make_srt(llm_translate, media_file=video_file)
  final_audio_path = await run_dubbing(srt_path, language, gender)
  dub_file=replace_audio_with_ffmpeg(video_file, final_audio_path)
  if remove_silence_from_video:
    final_path=remove_video_silences(dub_file, language)
    return final_path
  else:
    return dub_file




import gradio as gr

def ui1():
    with gr.Blocks() as demo:
        gr.Markdown("### 📜 Generate SRT, TXT, and Prompt from Audio/Video")

        with gr.Row():
            with gr.Column():
                media_input = gr.File(label="Upload Media File",value=None)
                lang_dropdown = gr.Dropdown(
                    choices=source_lang_list,
                    label="Video Language",
                    value="English"
                )
                target_lang = gr.Dropdown(
                    choices=target_language_list,
                    label="Dub Language",
                    value="Bengali"
                )
                run_button = gr.Button("Generate")
                # with gr.Accordion('Upload Video', open=False):
                #   media_input = gr.File(label="Upload Media File",value=None)

            with gr.Column():
                video_path=gr.Textbox(label="Video Path",show_copy_button=True)
                prompt_output = gr.Textbox(label="Generated Prompt", lines=4,show_copy_button=True)
                with gr.Accordion('🎬 Autoplay, Subtitle, Timestamp', open=False):
                    srt_output = gr.File(label="SRT Output")
                    txt_output = gr.File(label="TXT Output")

        run_button.click(
            fn=create_prompt,  # You must define this function elsewhere
            inputs=[media_input, lang_dropdown,target_lang],
            outputs=[srt_output, txt_output, prompt_output,video_path]
        )

    return demo




import asyncio
import json
import os




def run_video_edit(old_video, video_file, llm_translate, language, gender, remove_silence):
    # Validate video path
    if old_video and os.path.exists(old_video):
        vid_path = old_video
    elif video_file:
        if isinstance(video_file, str) and os.path.exists(video_file):
            vid_path = video_file
        elif hasattr(video_file, "name") and os.path.exists(video_file.name):
            vid_path = video_file.name
        else:
            gr.Warning("Uploaded video file is invalid.")
            return None
    else:
        gr.Warning("No video file provided.")
        return None

    # Parse JSON
    try:
        if isinstance(llm_translate, str):
            llm_translate = json.loads(llm_translate)
        elif not isinstance(llm_translate, dict):
            gr.Warning("Invalid translation format — must be JSON string or dict.")
            return None
    except json.JSONDecodeError as e:
        gr.Warning(f"Invalid JSON: {e}")
        return None

    # Run async dubbing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(video_edit(
            vid_path,
            llm_translate,
            language,
            gender,
            remove_silence_from_video=remove_silence
        ))
    finally:
        loop.close()

    return result




# # Wrap async function to run in sync context for Gradio
# def run_video_edit(old_video,video_file, llm_translate, language, gender, remove_silence):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     if os.path.exists(old_video):
#       vid_path=old_video
#     elif os.path.exists(video_file):
#       vid_path=video_file.name
#     else:
#       gr.Warning("Old Video Path not found, upload the video")
      
#     result = loop.run_until_complete(video_edit(
#         vid_path,
#         llm_translate,
#         language,
#         gender,
#         remove_silence_from_video=remove_silence
#     ))

#     return result

def ui2():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎬 AI Dubbing + Optional Silence Removal")

        with gr.Row():
            with gr.Column():
                old_video=gr.Textbox(label="Old Video Path")
                json_input = gr.Text(label="🧠 Paste Translated JSON",) 
                run_button = gr.Button("🚀 Run Dubbing")
                with gr.Accordion('🎬 Other setting', open=False):
                  
                  lang_dropdown = gr.Dropdown(
                      choices=target_language_list,
                      label="Language", value="Bengali"
                  )
                  gender_dropdown = gr.Dropdown(
                      choices=["Male", "Female"], label="Voice Gender", value="Female"
                  )
                  silence_toggle = gr.Checkbox(label="Remove Silence from Video", value=False)
                  video_input = gr.File(label="🎥 Upload Video", file_types=[".mp4"])#,value=old_video)
                
            
            with gr.Column():
                result_output = gr.File(label="📁 Dubbed Video Output")

        run_button.click(
            fn=run_video_edit,
            inputs=[old_video,video_input, json_input, lang_dropdown, gender_dropdown, silence_toggle],
            outputs=[result_output]
        )

    return demo


demo1 = ui1()
demo2 = ui2()
demo = gr.TabbedInterface([demo1, demo2],["Generate SRT with LLM prompt","Dub Video"],title="Video Dubbing")
demo.queue().launch(debug=True, share=True)
