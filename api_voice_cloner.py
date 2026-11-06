"""
API Voice Cloning Engine for Cambo Dubber & Python Applications.
Supports:
1. ElevenLabs API (Instant Voice Cloning with Audio Sample or Voice ID)
2. Custom Cloud / Khmer Website Voice Clone REST API (Multipart Audio + Text)
Includes robust error handling for Quota Exceeded, Invalid Key, Network Timeout, and Server Errors.
"""

import os
import sys
import time
import shutil
import subprocess
import hashlib
import requests
from pathlib import Path


def silent_subprocess_run(cmd, **kwargs):
    """Executes a subprocess completely silently without flashing a CMD window on Windows."""
    if sys.platform == "win32":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        if "startupinfo" not in kwargs:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
    return subprocess.run(cmd, **kwargs)



def merge_audio_chunks(chunk_paths: list, out_file: str, pause_ms: int = 200, sample_rate: int = 48000) -> str:
    """Merges audio chunks with a natural human breath pause between sentences."""
    import soundfile as sf
    import numpy as np
    from pathlib import Path

    loaded = []
    for p in chunk_paths:
        if not p or not os.path.exists(p) or os.path.getsize(p) < 100:
            continue
        data, sr = sf.read(p)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            dur = len(data) / float(sr)
            num_target = int(round(dur * sample_rate))
            orig_times = np.linspace(0, dur, len(data), endpoint=False)
            target_times = np.linspace(0, dur, num_target, endpoint=False)
            data = np.interp(target_times, orig_times, data).astype(np.float32)
        # Strip dead silence from chunk boundaries to prevent stutter/choppy speech
        t_data = trim_audio_edges(data.astype(np.float32), sample_rate, threshold=0.005, pad_ms=30)
        if len(t_data) > 0:
            loaded.append(t_data)

    if not loaded:
        raise RuntimeError("គ្មានឯកសារសំឡេងឃ្លាណាមួយត្រឹមត្រូវសម្រាប់ផ្គុំឡើយ")

    segments = []
    # Natural human sentence breath pause (140ms)
    effective_pause = min(pause_ms, 150)
    pause_samples = np.zeros(int((effective_pause / 1000.0) * sample_rate), dtype=np.float32)
    for i, d in enumerate(loaded):
        segments.append(d)
        if i < len(loaded) - 1:
            segments.append(pause_samples)

    buffer = np.concatenate(segments)
    max_val = np.max(np.abs(buffer))
    if max_val > 0.001:
        buffer = buffer * (0.95 / max_val)

    dest_p = Path(out_file).resolve()
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest_p), buffer, sample_rate)
    return str(dest_p)


class VoiceCloneAPIError(Exception):
    """Custom Exception for Voice Cloning API Errors with user-friendly messages."""
    pass



def trim_audio_edges(data, sr: int, threshold: float = 0.005, pad_ms: int = 30):
    """Trims dead silence from start and end of audio chunk with a gentle cushion."""
    import numpy as np
    frame_len = int(sr * 0.01) # 10ms frames
    if len(data) < frame_len * 2:
        return data

    start_idx = 0
    for i in range(0, len(data) - frame_len, frame_len):
        if np.sqrt(np.mean(data[i:i+frame_len]**2)) >= threshold:
            start_idx = max(0, i - int((pad_ms / 1000.0) * sr))
            break

    end_idx = len(data)
    for i in range(len(data) - frame_len, 0, -frame_len):
        if np.sqrt(np.mean(data[i:i+frame_len]**2)) >= threshold:
            end_idx = min(len(data), i + frame_len + int((pad_ms / 1000.0) * sr))
            break

    if start_idx < end_idx:
        return data[start_idx:end_idx]
    return data


def split_khmer_chunks(text: str, max_chars: int = 90) -> list:
    """
    Splits long Khmer text into natural, grammatical clauses/sentences to prevent
    autoregressive TTS models (VoxCPM2) from hallucinating or drifting into foreign accents.
    """
    import re
    if not text or not text.strip():
        return []

    clean = text.replace('\u200b', ' ').replace('\r', '').strip()

    primary_delims = re.split(r'([។?!;\n៖.:]+)', clean)
    sentences = []
    curr = ''
    for part in primary_delims:
        if re.match(r'^[។?!;\n៖.:]+$', part):
            curr += part
            if curr.strip():
                sentences.append(curr.strip())
            curr = ''
        else:
            curr += part
    if curr.strip():
        sentences.append(curr.strip())

    pass2_chunks = []
    for s in sentences:
        if len(s) <= max_chars:
            pass2_chunks.append(s)
        else:
            sub_parts = re.split(r'([ ,\t]+)', s)
            sub_curr = ''
            for sp in sub_parts:
                if len(sub_curr) + len(sp) > max_chars and sub_curr.strip():
                    pass2_chunks.append(sub_curr.strip())
                    sub_curr = sp
                else:
                    sub_curr += sp
            if sub_curr.strip():
                pass2_chunks.append(sub_curr.strip())

    final_chunks = []
    for c in pass2_chunks:
        if len(c) <= max_chars + 15:
            final_chunks.append(c)
        else:
            start = 0
            while start < len(c):
                chunk_slice = c[start:start + max_chars]
                if chunk_slice.strip():
                    final_chunks.append(chunk_slice.strip())
                start += max_chars

    return [c for c in final_chunks if c.strip()]



def clone_voice_elevenlabs_api(
    text: str,
    api_key: str,
    output_path: str,
    ref_audio_path: str = None,
    voice_id: str = None,
    stability: float = 0.5,
    similarity_boost: float = 0.85,
    emotion: str = "normal",
    speed: float = 1.0
) -> str:
    """
    Clones a voice and synthesizes speech using ElevenLabs API.
    
    Parameters:
    - text (str): The text to synthesize into speech.
    - api_key (str): ElevenLabs API Key (from elevenlabs.io).
    - output_path (str): Local path where the generated audio (.mp3 / .wav) will be saved.
    - ref_audio_path (str, optional): Path to reference audio sample (.mp3 / .wav) for Instant Voice Cloning.
    - voice_id (str, optional): Existing Voice ID if already cloned previously.
    - stability (float): Voice stability (0.0 to 1.0, default 0.5).
    - similarity_boost (float): Voice resemblance strength (0.0 to 1.0, default 0.85).

    Returns:
    - str: Absolute path to the saved audio file.
    """
    if not api_key or not api_key.strip():
        raise VoiceCloneAPIError("❌ មិនទាន់មាន ElevenLabs API Key! សូមបញ្ចូល API Key ជាមុនសិន។")

    if not text or not text.strip():
        raise VoiceCloneAPIError("❌ មិនមានអត្ថបទសម្រាប់និយាយ! សូមបញ្ចូលអត្ថបទ (Text)។")

    # STEP 1: Determine Voice ID or Clone from Reference Audio
    final_voice_id = voice_id.strip() if voice_id else None

    # If no Voice ID provided, but reference audio is provided, clone voice via /v1/voices/add
    if not final_voice_id:
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise VoiceCloneAPIError("❌ សូមជ្រើសរើសឯកសារសំឡេងគំរូ (Reference Audio) ឬបញ្ចូល Voice ID។")

        add_voice_url = "https://api.elevenlabs.io/v1/voices/add"
        headers_add = {
            "xi-api-key": api_key.strip()
        }
        voice_name = f"Clone_{Path(ref_audio_path).stem[:15]}"
        payload_data = {
            "name": voice_name,
            "description": "Auto cloned voice via Cambo Dubber Studio"
        }

        try:
            with open(ref_audio_path, "rb") as f_audio:
                files = {
                    "files": (Path(ref_audio_path).name, f_audio, "audio/mpeg")
                }
                res_clone = requests.post(
                    add_voice_url,
                    headers=headers_add,
                    data=payload_data,
                    files=files,
                    timeout=60
                )

            if res_clone.status_code == 200:
                clone_data = res_clone.json()
                final_voice_id = clone_data.get("voice_id")
            elif res_clone.status_code == 401:
                raise VoiceCloneAPIError("❌ API Key មិនត្រឹមត្រូវ (401 Unauthorized)! សូមពិនិត្យ ElevenLabs API Key ឡើងវិញ។")
            elif res_clone.status_code in [402, 429]:
                raise VoiceCloneAPIError("⚠️ អស់លុយ ឬអស់ Quota (429/402 Quota Exceeded)! គណនី ElevenLabs របស់អ្នកបានអស់កូតាសម្រាប់ Clone សំឡេងហើយ។")
            else:
                err_detail = res_clone.text
                try:
                    err_detail = res_clone.json().get("detail", {}).get("message", res_clone.text)
                except Exception:
                    pass
                raise VoiceCloneAPIError(f"❌ បរាជ័យក្នុងការ Clone សំឡេងពីឯកសារគំរូ ({res_clone.status_code}): {err_detail}")

        except requests.exceptions.ConnectionError:
            raise VoiceCloneAPIError("🌐 បញ្ហាអ៊ីនធឺណិត: មិនអាចភ្ជាប់ទៅកាន់ ElevenLabs API បានទេ! សូមពិនិត្យ Network របស់អ្នក។")
        except requests.exceptions.Timeout:
            raise VoiceCloneAPIError("⏳ អ៊ីនធឺណិតដើរយឺតពេក (Request Timeout): ការបញ្ជូនសំឡេងគំរូយូរជាង ៦០ វិនាទី។")

    if not final_voice_id:
        raise VoiceCloneAPIError("❌ មិនអាចទាញយក Voice ID សម្រាប់សំយោគសំឡេងបានទេ។")

    # STEP 2: Synthesize Text to Speech using the Cloned Voice ID
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{final_voice_id}"
    headers_tts = {
        "xi-api-key": api_key.strip(),
        "Content-Type": "application/json"
    }
    # Map emotion & style to ElevenLabs parameters
    style_boost = 0.0
    em_lower = (emotion or "normal").lower()
    if any(k in em_lower for k in ["news", "anchor", "reporter", "ព័ត៌មាន"]):
        stability = 0.82
        similarity_boost = 0.88
        style_boost = 0.05
    elif any(k in em_lower for k in ["storyteller", "narrator", "និទាន", "សម្រាយ"]):
        stability = 0.45
        similarity_boost = 0.80
        style_boost = 0.38
    elif any(k in em_lower for k in ["angry", "furious", "intense", "shout", "កាច", "ខឹង", "មោហោ", "ស្រែក"]):
        stability = 0.28
        similarity_boost = 0.75
        style_boost = 0.65
    elif any(k in em_lower for k in ["sad", "gentle", "soft", "កម្សត់", "ស្រទន់", "សុភាព"]):
        stability = 0.65
        similarity_boost = 0.80
        style_boost = 0.25
    elif any(k in em_lower for k in ["fast", "action", "ញាប់", "លឿន"]):
        stability = 0.35
        similarity_boost = 0.80
        style_boost = 0.40

    tts_body = {
        "text": text.strip(),
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity_boost),
            "style": float(style_boost),
            "use_speaker_boost": True
        }
    }

    try:
        res_tts = requests.post(tts_url, headers=headers_tts, json=tts_body, timeout=60)

        if res_tts.status_code == 200:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f_out:
                f_out.write(res_tts.content)
            final_audio = str(Path(output_path).resolve())
            if (emotion and emotion != "normal") or (speed and abs(float(speed) - 1.0) >= 0.03):
                final_audio = apply_audio_emotion_and_speed(final_audio, final_audio, speed=speed, emotion=emotion)
            return final_audio

        elif res_tts.status_code == 401:
            raise VoiceCloneAPIError("❌ API Key មិនត្រឹមត្រូវ (401 Unauthorized)! សូមពិនិត្យ ElevenLabs API Key។")
        elif res_tts.status_code in [402, 429]:
            raise VoiceCloneAPIError("⚠️ អស់លុយ ឬអស់ Quota (429/402 Quota Exceeded)! គណនី ElevenLabs របស់អ្នកបានអស់ Characters សម្រាប់បង្កើតសំឡេងហើយ។")
        else:
            err_detail = res_tts.text
            try:
                err_detail = res_tts.json().get("detail", {}).get("message", res_tts.text)
            except Exception:
                pass
            raise VoiceCloneAPIError(f"❌ ElevenLabs TTS Error ({res_tts.status_code}): {err_detail}")

    except requests.exceptions.ConnectionError:
        raise VoiceCloneAPIError("🌐 បញ្ហាអ៊ីនធឺណិត: ដាច់ការតភ្ជាប់ទៅកាន់ ElevenLabs API!")
    except requests.exceptions.Timeout:
        raise VoiceCloneAPIError("⏳ Request Timeout: ការបង្កើតសំឡេងស៊ីពេលយូរជាង ៦០ វិនាទី។")


def clone_voice_custom_cloud_api(
    text: str,
    ref_audio_path: str,
    api_endpoint: str,
    api_key: str,
    output_path: str,
    custom_headers: dict = None,
    timeout_seconds: int = 90
) -> str:
    """
    Clones voice and synthesizes speech using any generic Khmer / Cloud REST API Endpoint.
    Sends reference audio file as multipart/form-data and text in form data or headers.

    Parameters:
    - text (str): The text to synthesize.
    - ref_audio_path (str): Local path to reference audio file.
    - api_endpoint (str): The full URL endpoint of the Voice Cloning service.
    - api_key (str): Bearer token or API key for authentication.
    - output_path (str): Local destination file path (.mp3 or .wav).
    - custom_headers (dict, optional): Custom HTTP headers.
    - timeout_seconds (int): Network timeout in seconds.

    Returns:
    - str: Absolute path to saved audio file.
    """
    if not api_endpoint or not api_endpoint.startswith("http"):
        raise VoiceCloneAPIError("❌ Endpoint URL មិនត្រឹមត្រូវ! សូមបញ្ចូល URL ដូចជា https://example.com/api/v1/voice-clone")

    if not os.path.exists(ref_audio_path):
        raise VoiceCloneAPIError(f"❌ រកមិនឃើញឯកសារសំឡេងគំរូ: {ref_audio_path}")

    headers = custom_headers or {}
    if api_key and api_key.strip():
        # Standard Bearer or API-Key header
        headers["Authorization"] = f"Bearer {api_key.strip()}"
        headers["x-api-key"] = api_key.strip()

    data_payload = {
        "text": text.strip()
    }

    try:
        with open(ref_audio_path, "rb") as f_ref:
            files = {
                "audio": (Path(ref_audio_path).name, f_ref, "audio/wav")
            }

            resp = requests.post(
                api_endpoint,
                headers=headers,
                data=data_payload,
                files=files,
                timeout=timeout_seconds
            )

        if resp.status_code == 200:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f_out:
                f_out.write(resp.content)
            return str(Path(output_path).resolve())

        elif resp.status_code == 401:
            raise VoiceCloneAPIError("❌ API Key មិនត្រឹមត្រូវ ឬគ្មានសិទ្ធិប្រើប្រាស់ (401 Unauthorized)!")
        elif resp.status_code in [402, 429]:
            raise VoiceCloneAPIError("⚠️ អស់លុយ ឬអស់ Quota (429 Too Many Requests / 402 Payment Required) លើប្រព័ន្ធ API នោះ!")
        elif resp.status_code >= 500:
            raise VoiceCloneAPIError(f"💥 Website Server កំពុងមានបញ្ហា (HTTP {resp.status_code}): Server Error")
        else:
            raise VoiceCloneAPIError(f"❌ API Error ({resp.status_code}): {resp.text[:300]}")

    except requests.exceptions.ConnectionError:
        raise VoiceCloneAPIError("🌐 មិនអាចភ្ជាប់ទៅកាន់ Website API នោះបានទេ! សូមពិនិត្យ Internet ឬ URL របស់ Endpoint។")
    except requests.exceptions.Timeout:
        raise VoiceCloneAPIError("⏳ Request Timeout: Website មិនបានឆ្លើយតបក្នុងរយៈពេលកំណត់។")


# =========================================================================
# ៣. FUNCTION សម្រាប់ KIRITTS API (ជាមួយ MD5 Caching, Retry 429, និង Credit Check)
# =========================================================================

# -------------------------------------------------------------------------
# [KIRITTS API] KiriTTS គឺជាប្រព័ន្ធ OpenAI-Compatible REST API
# Endpoint ស្តង់ដារ: https://api.kiritts.com/v1/audio/speech
# -------------------------------------------------------------------------
KIRITTS_DEFAULT_ENDPOINT = "https://api.kiritts.com/v1/audio/speech"


def apply_audio_emotion_and_speed(input_audio: str, output_audio: str, speed: float = 1.0, emotion: str = "normal") -> str:
    """
    Applies audio speed pacing (atempo) and emotional acoustic shaping (soft/gentle vs aggressive/angry)
    using FFmpeg.
    """
    if not input_audio or not os.path.exists(input_audio):
        return input_audio

    filters = []

    # 1. Speed pacing (atempo between 0.5x and 2.0x)
    if speed and abs(float(speed) - 1.0) >= 0.03:
        spd = max(0.5, min(2.0, float(speed)))
        filters.append(f"atempo={spd:.2f}")

    # 2. Emotional tone & style shaping (Storyteller, News Anchor, Angry, Fast, Sad, Fear)
    em = (emotion or "normal").lower()
    if any(k in em for k in ["storyteller", "narrator", "និទាន", "សម្រាយ"]):
        # Warm narrative presence: warm low-end boost (180Hz), slight soften of harsh mids, studio broadcast compression
        filters.append("equalizer=f=180:t=q:w=1.2:g=3.5")
        filters.append("equalizer=f=2800:t=q:w=1.5:g=-1.5")
        filters.append("acompressor=threshold=-18dB:ratio=2.5:attack=10:release=80")
        filters.append("volume=1.05")
    elif any(k in em for k in ["news", "anchor", "reporter", "ព័ត៌មាន"]):
        # Crisp news anchor: highpass 85Hz, crisp high-mid presence boost at 3500Hz for razor-sharp consonants, broadcast compand
        filters.append("highpass=f=85")
        filters.append("equalizer=f=3500:t=q:w=1.5:g=3.5")
        filters.append("acompressor=threshold=-16dB:ratio=3.2:attack=4:release=40")
        filters.append("volume=1.08")
    elif any(k in em for k in ["angry", "furious", "intense", "shout", "កាច", "ខឹង", "មោហោ", "ស្រែក"]):
        # Aggressive, punchy, forward, intense upper-mid punch at 2600Hz + high gain
        filters.append("highpass=f=110")
        filters.append("equalizer=f=2600:t=q:w=1.0:g=4.5")
        filters.append("acompressor=threshold=-12dB:ratio=4:attack=2:release=25")
        filters.append("volume=1.22")
    elif any(k in em for k in ["fast", "action", "ញាប់", "លឿន", "រំភើប"]):
        # Energetic and fast: bright high-mid boost, energetic presence
        filters.append("highpass=f=100")
        filters.append("equalizer=f=3200:t=q:w=1.2:g=2.8")
        filters.append("volume=1.12")
    elif any(k in em for k in ["sad", "gentle", "soft", "sorrowful", "កម្សត់", "ស្រទន់", "សុភាព", "អាណិត"]):
        # Soft, slow, emotional, lowpass to remove harshness
        filters.append("lowpass=f=6200")
        filters.append("equalizer=f=300:t=q:w=1.0:g=1.5")
        filters.append("volume=0.95")
    elif any(k in em for k in ["fear", "scared", "horror", "ភ័យ", "តក់ស្លុត"]):
        # Fearful, tense, airy highs
        filters.append("highpass=f=150")
        filters.append("equalizer=f=4500:t=q:w=1.5:g=3.2")
        filters.append("volume=0.96")

    if not filters:
        if input_audio != output_audio:
            shutil.copyfile(input_audio, output_audio)
        return output_audio

    try:
        import imageio_ffmpeg
        FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_EXE = "ffmpeg"

    ext = Path(output_audio).suffix or ".wav"
    temp_out = str(Path(output_audio).with_name(f"temp_emo_{int(time.time()*1000)}{ext}"))
    filter_str = ",".join(filters)

    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_audio,
        "-filter:a", filter_str
    ]
    if ext.lower() == ".wav":
        cmd.extend(["-c:a", "pcm_s16le", "-ar", "48000"])
    cmd.append(temp_out)

    try:
        silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            if os.path.exists(output_audio):
                try:
                    os.remove(output_audio)
                except Exception:
                    pass
            shutil.move(temp_out, output_audio)
            return output_audio
    except Exception:
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except Exception:
                pass

    if input_audio != output_audio:
        shutil.copyfile(input_audio, output_audio)
    return output_audio


def synthesize_kiritts_api(
    text: str,
    api_key: str,
    voice_id: str = "Maly",
    ref_audio_path: str = None,
    output_path: str = None,
    endpoint_url: str = KIRITTS_DEFAULT_ENDPOINT,
    cache_dir: str = "cache_audio/kiritts",
    max_retries: int = 3,
    retry_delay_seconds: float = 2.5,
    speed: float = 1.0,
    emotion: str = "normal",
    style_prompt: str = ""
) -> str:
    """
    បង្កើតសំឡេងនិយាយ ឬ Clone សំឡេងតាមរយៈ KiriTTS API (OpenAI Compatible)
    ជាមួយប្រព័ន្ធ MD5 Caching, Emotion Control និង Speed Rate។
    """
    import hashlib
    import time
    import shutil

    if not api_key or not api_key.strip():
        raise VoiceCloneAPIError("❌ មិនទាន់មាន KiriTTS API Key! សូមបញ្ចូល API Key ជាមុនសិន។")

    clean_text = text.strip() if text else ""
    if not clean_text:
        raise VoiceCloneAPIError("❌ អត្ថបទមិនអាចទទេបានឡើយ! សូមបញ្ចូលអត្ថបទ (Text)។")

    # ជ្រើសរើស Voice (សម្អាត Text Label ប្រសិនបើមាន "Bora (ប្រុស)" ឬ "🎙️ ...")
    active_voice = "Maly"
    if voice_id and voice_id.strip():
        v_raw = voice_id.strip()
        if "Arun_villain" in v_raw:
            active_voice = "Arun"
            emotion = "angry"
        elif "Arun_elder" in v_raw:
            active_voice = "Arun"
            speed = 0.85
        elif "Chanda_young" in v_raw:
            active_voice = "Chanda"
            emotion = "gentle"
        elif v_raw.startswith("Maly"):
            active_voice = "Maly"
        elif v_raw.startswith("Bora"):
            active_voice = "Bora"
        elif v_raw.startswith("Chanda"):
            active_voice = "Chanda"
        elif v_raw.startswith("Arun"):
            active_voice = "Arun"
        elif v_raw.startswith("🎙️"):
            active_voice = v_raw.replace("🎙️", "").strip()
        elif "Auto Clone" in v_raw or v_raw == "default":
            active_voice = ""
        else:
            active_voice = v_raw

    # ប្រសិនបើមិនបានបញ្ជាក់ Voice ណាមួយ ហើយមាន Reference Audio ពីកុំព្យូទ័រ ធ្វើការ Upload ទៅ Clone សំឡេងថ្មីលើ KiriTTS
    if not active_voice and ref_audio_path and os.path.exists(ref_audio_path):
        try:
            clone_url = "https://api.kiritts.com/v1/audio/voice-clones"
            voice_name = f"Clone_{Path(ref_audio_path).stem[:12]}"
            with open(ref_audio_path, "rb") as f_ref:
                files = {"file": (Path(ref_audio_path).name, f_ref, "audio/mpeg")}
                r_c = requests.post(
                    clone_url,
                    headers={"Authorization": f"Bearer {api_key.strip()}"},
                    data={"name": voice_name},
                    files=files,
                    timeout=60
                )
                if r_c.status_code in [200, 201]:
                    c_data = r_c.json()
                    active_voice = c_data.get("name") or c_data.get("id") or voice_name
        except Exception:
            pass  # បើបរាជ័យក្នុងការ clone សំឡេងថ្មី គឺប្រើ Maly ធម្មតា

    if not active_voice:
        active_voice = "Maly"

    # ---------------------------------------------------------------------
    # ជំហានទី ១: ប្រព័ន្ធ MD5 Caching (ឆែកមើលថាតើធ្លាប់ Generate រួចហើយឬនៅ)
    # ---------------------------------------------------------------------
    cache_signature = f"{clean_text}_{active_voice}_{round(float(speed), 2)}_{emotion or ''}_{style_prompt or ''}".encode("utf-8")
    md5_hash = hashlib.md5(cache_signature).hexdigest()

    cache_folder = Path(cache_dir)
    cache_folder.mkdir(parents=True, exist_ok=True)
    cached_file_path = cache_folder / f"kiri_{md5_hash}.mp3"

    final_output = output_path or str(cached_file_path)
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)

    if cached_file_path.exists() and cached_file_path.stat().st_size > 0:
        if final_output != str(cached_file_path):
            shutil.copyfile(str(cached_file_path), final_output)
            apply_audio_emotion_and_speed(final_output, final_output, speed=speed, emotion=emotion)
            return str(Path(final_output).resolve())
        return str(cached_file_path.resolve())

    # ---------------------------------------------------------------------
    # ជំហានទី ២: រៀបចំ Headers និង Payload តាមស្តង់ដារ KiriTTS (OpenAI Format)
    # ---------------------------------------------------------------------
    # ធានាថា URL មិនមែនជា URL 404 ចាស់
    clean_endpoint = endpoint_url.strip()
    if clean_endpoint.endswith("/v1/synthesize"):
        clean_endpoint = "https://api.kiritts.com/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg, audio/wav, */*"
    }

    # Payload តាមស្តង់ដារ OpenAI TTS ដែល KiriTTS ប្រើប្រាស់
    payload = {
        "model": "kiritts",
        "input": clean_text,
        "voice": active_voice,
        "response_format": "mp3"
    }
    if speed and abs(float(speed) - 1.0) >= 0.03:
        payload["speed"] = round(float(speed), 2)

    # ---------------------------------------------------------------------
    # ជំហានទី ៣: បញ្ជូន POST Request ជាមួយ Retry Loop សម្រាប់ Error 429
    # ---------------------------------------------------------------------
    last_error_msg = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                clean_endpoint,
                headers=headers,
                json=payload,
                timeout=90
            )

            # -------------------------------------------------------------
            # ជំហានទី ៤: វិភាគ Response និងចាប់កំហុស (Status Codes)
            # -------------------------------------------------------------
            if resp.status_code == 200:
                with open(cached_file_path, "wb") as f_cache:
                    f_cache.write(resp.content)

                if final_output != str(cached_file_path):
                    with open(final_output, "wb") as f_out:
                        f_out.write(resp.content)

                apply_audio_emotion_and_speed(final_output, final_output, speed=speed, emotion=emotion)
                return str(Path(final_output).resolve())

            # ២. ករណី Error 429: Rate Limit Exceeded (ហៅញឹកញាប់ពេក)
            elif resp.status_code == 429:
                wait_time = retry_delay_seconds * attempt
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue  # Retry ម្តងទៀត
                else:
                    raise VoiceCloneAPIError(
                        f"⚠️ KiriTTS API: លើសកម្រិតកំណត់នៃការហៅ (429 Rate Limit Exceeded)! "
                        f"បានព្យាយាម {max_retries} ដងរួចហើយ។ សូមរង់ចាំបន្តិចសិនមុននឹងសាកល្បងម្ដងទៀត។"
                    )

            # ៣. ករណី Error 402 / 403: អស់ Credit ឬ Quota
            elif resp.status_code in [402, 403]:
                err_text = resp.text
                raise VoiceCloneAPIError(
                    f"💳 KiriTTS API: អស់ Credit ឬ Quota ក្នុងការប្រើប្រាស់ហើយ (HTTP {resp.status_code})! "
                    f"សូមពិនិត្យមើលសមតុល្យទឹកប្រាក់ក្នុង Dashboard របស់អ្នក។ ព័ត៌មានលម្អិត: {err_text[:150]}"
                )

            # ៤. ករណី Error 401: API Key មិនត្រឹមត្រូវ
            elif resp.status_code == 401:
                raise VoiceCloneAPIError(
                    "❌ KiriTTS API Key មិនត្រឹមត្រូវ (401 Unauthorized)! សូមពិនិត្យ API Key របស់អ្នកឡើងវិញ។"
                )

            # ៥. ករណី Server Error (HTTP 500/502/503)
            elif resp.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds)
                    continue
                raise VoiceCloneAPIError(
                    f"💥 KiriTTS Server កំពុងរវល់ ឬមានបញ្ហា (HTTP {resp.status_code})! សូមសាកល្បងម្ដងទៀតបន្តិចក្រោយ។"
                )

            # ៦. កំហុសផ្សេងៗ
            else:
                raise VoiceCloneAPIError(f"❌ KiriTTS API Error (HTTP {resp.status_code}): {resp.text[:250]}")

        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)
                continue
            raise VoiceCloneAPIError("🌐 បញ្ហាអ៊ីនធឺណិត: មិនអាចភ្ជាប់ទៅកាន់ KiriTTS API បានទេ! សូមពិនិត្យ Network ឬ Endpoint URL។")
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)
                continue
            raise VoiceCloneAPIError("⏳ Request Timeout: ការហៅទៅកាន់ KiriTTS API លើសពី ៩០ វិនាទី។")

    raise VoiceCloneAPIError(f"❌ មិនអាចទាញយកសំឡេងពី KiriTTS API បានទេ: {last_error_msg}")


def fetch_kiritts_voices(api_key: str) -> list:
    """
    ទាញយកបញ្ជីសំឡេងទាំងអស់ (ទាំងសំឡេងដើម និងសំឡេង Cloned ផ្ទាល់ខ្លួន) ពីគណនី KiriTTS តាមរយៈ API។
    """
    voices = []
    if not api_key or not api_key.strip():
        return voices

    headers = {"Authorization": f"Bearer {api_key.strip()}"}

    # 1. ពិនិត្យបញ្ជីសំឡេង Cloned: GET /v1/audio/voice-clones
    try:
        r1 = requests.get("https://api.kiritts.com/v1/audio/voice-clones", headers=headers, timeout=8)
        if r1.status_code == 200:
            data = r1.json()
            items = data if isinstance(data, list) else data.get("data", data.get("voices", []))
            for item in items:
                v_name = item.get("name") or item.get("id") if isinstance(item, dict) else str(item)
                if v_name and v_name not in voices:
                    voices.append(v_name)
    except Exception:
        pass

    # 2. ពិនិត្យបញ្ជីសំឡេងទូទៅ: GET /v1/voices
    try:
        r2 = requests.get("https://api.kiritts.com/v1/voices", headers=headers, timeout=8)
        if r2.status_code == 200:
            data = r2.json()
            items = data if isinstance(data, list) else data.get("data", data.get("voices", []))
            for item in items:
                v_name = item.get("name") or item.get("id") if isinstance(item, dict) else str(item)
                if v_name and v_name not in voices:
                    voices.append(v_name)
    except Exception:
        pass

    return voices


# =========================================================================
# ៤. FUNCTION សម្រាប់ GOOGLE COLAB GPU SERVER (VOXCPM2 + NGROK)
# =========================================================================

def sanitize_colab_url(raw_url: str) -> str:
    """Sanitizes and auto-fixes malformed Colab / Cloudflare / Ngrok URLs (e.g. missing 'htt', spaces)."""
    if not raw_url:
        return ""
    u = raw_url.strip().rstrip("/")
    if u.startswith("ps://"):
        u = "https://" + u[5:]
    elif u.startswith("tps://"):
        u = "https://" + u[6:]
    elif u.startswith("ttps://"):
        u = "https://" + u[7:]
    elif not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u

    # Replace spaces in domain with hyphens (Cloudflare tunnel slug format)
    parts = u.split("://", 1)
    if len(parts) == 2:
        scheme, rest = parts
        domain_and_path = rest.split("/", 1)
        domain = domain_and_path[0].replace(" ", "-")
        path = ("/" + domain_and_path[1]) if len(domain_and_path) > 1 else ""
        u = f"{scheme}://{domain}{path}"
    return u


def ping_colab_server(colab_url: str, timeout: int = 10) -> dict:
    """
    ពិនិត្យការតភ្ជាប់ទៅកាន់ Google Colab Server តាមរយៈ Ngrok URL។
    Return ព័ត៌មានស្ថានភាព Server, Device និង GPU។
    """
    if not colab_url or not colab_url.strip():
        raise VoiceCloneAPIError("❌ មិនទាន់មាន Colab Public URL ទេ! សូមបញ្ចូល Ngrok URL ជាមុនសិន។")

    clean_url = sanitize_colab_url(colab_url)

    health_url = f"{clean_url}/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "online": True,
                "device": data.get("device", "cuda"),
                "gpu": data.get("gpu", "NVIDIA GPU"),
                "ready": data.get("ready", True)
            }
        else:
            raise VoiceCloneAPIError(
                f"⚠️ Colab Server បានឆ្លើយតបកូដ {resp.status_code} ប៉ុន្តែមិនទាន់រួចរាល់ទេ: {resp.text[:100]}"
            )
    except requests.exceptions.ConnectionError:
        raise VoiceCloneAPIError(
            "🌐 មិនអាចភ្ជាប់ទៅកាន់ Google Colab បានទេ!\n\n"
            "មូលហេតុដែលអាចកើតមាន:\n"
            "១. Notebook លើ Google Colab មិនទាន់ចុច Run ឬដាច់ការតភ្ជាប់ (Disconnect)\n"
            "២. Ngrok Public URL ផុតកំណត់ ឬវាយខុស (សូមចម្លង Link ពី Colab មកដាក់ម្តងទៀត)\n"
            "៣. កុំព្យូទ័ររបស់អ្នកដាច់ Internet"
        )
    except requests.exceptions.Timeout:
        raise VoiceCloneAPIError(
            "⏳ Colab Server ឆ្លើយតបយឺតលើសពី ១០ វិនាទី (Timeout)! "
            "សូមពិនិត្យមើល Colab ថាតើកំពុង Run កូដ ឬកំពុងផ្ទុក Model នៅឡើយ។"
        )
    except Exception as e:
        raise VoiceCloneAPIError(f"❌ បរាជ័យក្នុងការតភ្ជាប់ទៅកាន់ Colab Server: {str(e)}")


def clone_voice_colab_api(
    text: str,
    colab_url: str,
    output_path: str,
    ref_audio_path: str = None,
    prompt_text: str = None,
    cfg_value: float = 2.0,
    timesteps: int = 10,
    speed: float = 1.0,
    emotion: str = "normal",
    cache_dir: str = "cache_audio/colab_voxcpm",
    timeout: int = 150
) -> str:
    """
    បញ្ជូនអត្ថបទ និងសំឡេងគំរូទៅកាន់ Google Colab GPU Server តាមរយៈ Public Ngrok URL
    ដើម្បី Clone សំឡេងដោយប្រើ VoxCPM2។

    លក្ខណៈពិសេស:
    1. ប្រព័ន្ធ MD5 Caching ឆ្លាតវៃ: មិនហៅ API ជាន់គ្នាលើប្រយោគដែលធ្លាប់ Clone រួចឡើយ (សន្សំពេល & លឿន)
    2. Try/Except ចាប់ Error គ្រប់ជ្រុងជ្រោយ (ConnectionError, Timeout, Server Crash)
    3. Emotion & Speed Post-Processing តាមតម្រូវការ
    """
    import hashlib
    import shutil

    if not colab_url or not colab_url.strip():
        raise VoiceCloneAPIError("❌ សូមបញ្ចូល Google Colab Ngrok URL ជាមុនសិន!")

    clean_text = text.strip() if text else ""
    if not clean_text:
        raise VoiceCloneAPIError("❌ មិនមានអត្ថបទសម្រាប់និយាយទេ! សូមបញ្ចូលអត្ថបទ (Text)។")

    # Anti-hallucination auto-chunking for text longer than 35 characters
    if len(clean_text) > 90:
        chunks = split_khmer_chunks(clean_text, max_chars=90)
        if len(chunks) > 1:
            chunk_files = []
            for idx, c_text in enumerate(chunks):
                c_out = str(Path(cache_dir) / f"subcolab_{idx}_{hashlib.md5((c_text + str(ref_audio_path)).encode('utf-8')).hexdigest()[:8]}.wav")
                c_res = clone_voice_colab_api(
                    text=c_text,
                    colab_url=colab_url,
                    output_path=c_out,
                    ref_audio_path=ref_audio_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    timesteps=timesteps,
                    speed=1.0,
                    emotion="normal",
                    cache_dir=cache_dir,
                    timeout=timeout
                )
                if c_res and os.path.exists(c_res) and os.path.getsize(c_res) > 300:
                    chunk_files.append(c_res)
            if chunk_files:
                final_merged = merge_audio_chunks(chunk_files, output_path, pause_ms=140, sample_rate=48000)
                return apply_audio_emotion_and_speed(final_merged, final_merged, speed=speed, emotion=emotion)

    clean_url = sanitize_colab_url(colab_url)

    endpoint_url = f"{clean_url}/clone"


    # -------------------------------------------------------------------------
    # ជំហានទី ១: MD5 Caching System (ការពារកុំឱ្យហៅ API ជាន់គ្នា)
    # -------------------------------------------------------------------------
    ref_signature = ""
    if ref_audio_path and os.path.exists(ref_audio_path):
        try:
            # យក Size និង Modified Time ធ្វើជា Fingerprint នៃឯកសារសំឡេងគំរូ
            ref_stat = os.stat(ref_audio_path)
            ref_signature = f"{ref_stat.st_size}_{int(ref_stat.st_mtime)}"
        except Exception:
            ref_signature = str(Path(ref_audio_path).name)

    cache_str = f"colab_voxcpm_{clean_text}_{ref_signature}_{round(float(cfg_value), 2)}_{timesteps}_{prompt_text or ''}_{round(float(speed), 2)}_{emotion or ''}"
    md5_hash = hashlib.md5(cache_str.encode("utf-8")).hexdigest()

    cache_folder = Path(cache_dir)
    cache_folder.mkdir(parents=True, exist_ok=True)
    cached_file_path = cache_folder / f"vox_{md5_hash}.wav"

    final_output = output_path or str(cached_file_path)
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)

    # បើធ្លាប់ Clone រួចហើយ និងមាន File ក្នុង Cache: យកមកប្រើភ្លាមៗ!
    if cached_file_path.exists() and cached_file_path.stat().st_size > 1000:
        if final_output != str(cached_file_path):
            shutil.copyfile(str(cached_file_path), final_output)
            apply_audio_emotion_and_speed(final_output, final_output, speed=speed, emotion=emotion)
            return str(Path(final_output).resolve())
        return str(cached_file_path.resolve())

    # -------------------------------------------------------------------------
    # ជំហានទី ២: ផ្ញើ Request ទៅកាន់ Google Colab Server តាម requests
    # -------------------------------------------------------------------------
    payload_data = {
        "text": clean_text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(timesteps)
    }
    if prompt_text and prompt_text.strip():
        payload_data["prompt_text"] = prompt_text.strip()

    try:
        files = {}
        file_handle = None
        upload_path = ref_audio_path
        trimmed_temp = None
        if ref_audio_path and os.path.exists(ref_audio_path):
            try:
                import imageio_ffmpeg
                ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
                trimmed_temp = str(cache_folder / f"ref_trimmed_{int(time.time()*1000)}.wav")
                cmd_trim = [ff_exe, "-y", "-i", ref_audio_path, "-t", "6.5", "-ac", "1", "-ar", "16000", trimmed_temp]
                r_trim = silent_subprocess_run(cmd_trim, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if r_trim.returncode == 0 and os.path.exists(trimmed_temp) and os.path.getsize(trimmed_temp) > 1000:
                    upload_path = trimmed_temp
            except Exception:
                upload_path = ref_audio_path

            file_handle = open(upload_path, "rb")
            files = {
                "ref_audio": (Path(upload_path).name, file_handle, "audio/wav")
            }

        resp = None
        for attempt in range(2):
            try:
                if files and file_handle and file_handle.closed:
                    file_handle = open(upload_path, "rb")
                    files["ref_audio"] = (Path(upload_path).name, file_handle, "audio/wav")
                resp = requests.post(
                    endpoint_url,
                    data=payload_data,
                    files=files if files else None,
                    timeout=timeout
                )
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                raise net_err
            finally:
                if file_handle and not file_handle.closed and attempt == 1:
                    file_handle.close()

        if file_handle and not file_handle.closed:
            file_handle.close()

        # ---------------------------------------------------------------------
        # ជំហានទី ៣: ពិនិត្យលទ្ធផលឆ្លើយតប និងចាប់ Error
        # ---------------------------------------------------------------------
        if resp.status_code == 200:
            # រក្សាទុកចូល Cache និងឯកសារគោលដៅ
            with open(str(cached_file_path), "wb") as f_cache:
                f_cache.write(resp.content)

            if final_output != str(cached_file_path):
                shutil.copyfile(str(cached_file_path), final_output)

            # កែសម្រួល Emotion និងល្បឿន Speed ប្រសិនបើមាន
            apply_audio_emotion_and_speed(final_output, final_output, speed=speed, emotion=emotion)
            return str(Path(final_output).resolve())

        elif resp.status_code == 400:
            err_msg = "Bad Request"
            try:
                err_msg = resp.json().get("error", resp.text)
            except Exception:
                pass
            raise VoiceCloneAPIError(f"❌ Colab ស្នើសុំទិន្នន័យខុស (400): {err_msg}")

        elif resp.status_code >= 500:
            err_msg = "Server Error"
            try:
                err_msg = resp.json().get("error", resp.text)
            except Exception:
                pass
            raise VoiceCloneAPIError(
                f"💥 Google Colab Server មានបញ្ហា (HTTP {resp.status_code}): {err_msg}\n"
                f"សូមពិនិត្យមើលផ្ទាំង Output ក្នុង Google Colab ថាតើមានកំហុសអ្វីកើតឡើង។"
            )
        else:
            raise VoiceCloneAPIError(f"❌ Colab API Error ({resp.status_code}): {resp.text[:200]}")

    except requests.exceptions.ConnectionError:
        raise VoiceCloneAPIError(
            "🌐 បញ្ហាដាច់ការតភ្ជាប់: មិនអាចភ្ជាប់ទៅកាន់ Google Colab Server បានទេ!\n\n"
            "សូមពិនិត្យមើល:\n"
            "១. Google Colab អាចដាច់ Connection ឬ Sleep (សូមបើកមើលផ្ទាំង Colab លើ Browser)\n"
            "២. Ngrok Public URL ត្រឹមត្រូវដែរឬទេ?\n"
            "៣. ពិនិត្យមើលថាតើ Cell លើ Colab កំពុង Run ជាប់ជានិច្ចដែរឬទេ"
        )
    except requests.exceptions.Timeout:
        raise VoiceCloneAPIError(
            f"⏳ Request Timeout: Google Colab ស៊ីពេលយូរជាង {timeout} វិនាទី!\n"
            "អាចមកពីអត្ថបទវែងពេក ឬ Colab កំពុងដំណើរការការងារធ្ងន់។ សូមសាកល្បងម្ដងទៀត។"
        )


