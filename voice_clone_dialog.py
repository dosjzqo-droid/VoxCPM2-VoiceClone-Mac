"""
Voice Cloning Studio for Cambo Dubber (Csophouret Suite).
Powered by OpenBMB VoxCPM2 Foundation Model & Smart Neural Voice Matching.
Provides:
1. Reference Audio 16kHz Resampling & Feature Extraction.
2. Zero-Shot Studio-Quality 48kHz Voice Cloning with VoxCPM2.
3. Fast Neural Voice Matching (Instant 1s fallback mode).
4. Fast Subprocess FFmpeg Video Muxing (-c:v copy).
5. Modern GUI Dialog with live playback, video audio extraction, and subtitle table integration.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# Pre-initialize PyTorch runtime before PyQt6 to avoid Windows DLL WinError 1114 collision with Qt
try:
    import torch
except Exception:
    pass

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QRadioButton, QButtonGroup, QGroupBox, QSlider, QComboBox,
    QFrame, QWidget, QScrollArea, QApplication,
    QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
    QCheckBox, QSizePolicy
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


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

if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent

CACHE_DIR = (APP_ROOT / "cache_audio").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = (APP_ROOT / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_system_sound_path(filename: str) -> str:
    """Resolves path to bundled default system sound files."""
    candidates = [
        APP_ROOT / "assets" / "system_sounds" / filename,
        APP_ROOT / "_internal" / "assets" / "system_sounds" / filename,
        Path(sys.executable).resolve().parent / "assets" / "system_sounds" / filename,
        Path(sys.executable).resolve().parent / "_internal" / "assets" / "system_sounds" / filename,
        Path("D:/DramaCoder/assets/system_sounds") / filename,
        Path("D:/DramaCoder/dist/Systems sound") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return ""

from icons import get_google_icon


_VOXCPM_READY_CACHE = None


def is_voxcpm_ready() -> bool:
    """Fast check if voxcpm is installed without importing heavy AI weights on the UI thread."""
    global _VOXCPM_READY_CACHE
    if _VOXCPM_READY_CACHE is not None:
        return _VOXCPM_READY_CACHE
    import importlib.util
    _VOXCPM_READY_CACHE = (importlib.util.find_spec("voxcpm") is not None)
    return _VOXCPM_READY_CACHE



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


def preprocess_reference_audio_16k(input_audio_path: str, temp_output_16k: str) -> str:
    """
    Resamples reference audio to 16kHz mono using FFmpeg.
    """
    if not input_audio_path or not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"រកមិនឃើញឯកសារសំឡេងគំរូ: {input_audio_path}")
    
    Path(temp_output_16k).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_audio_path,
        "-t", "6.5",
        "-ac", "1",
        "-ar", "16000",
        temp_output_16k
    ]
    res = silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError("បរាជ័យក្នុងការបម្លែងសំឡេងគំរូទៅជា 16kHz")
    return temp_output_16k


_CACHED_VOXCPM_MODEL = None
_CACHED_MODEL_KEY = None


def clone_voice_voxcpm2_core(
    text: str,
    prompt_audio_16k: str,
    output_wav_path: str,
    prompt_text: str = None,
    model_id_or_path: str = "openbmb/VoxCPM2",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    device: str = "auto",
    status_callback = None
) -> str:
    """
    Core Voice Cloning execution using VoxCPM2 with in-memory caching.
    """
    global _CACHED_VOXCPM_MODEL, _CACHED_MODEL_KEY
    try:
        from voxcpm import VoxCPM
        import soundfile as sf
        import torch
    except ImportError:
        raise ImportError(
            "សូមដំឡើង VoxCPM ជាមុនសិនដោយដំណើរការ Install_VoxCPM_Dependencies.bat"
        )

    # Resolve safe device (GTX 1650 with 4GB VRAM needs CPU to avoid CUDA OOM for 2B params)
    actual_device = device
    if device in ["auto", None]:
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 7 * (1024**3):
            actual_device = "cuda"
        else:
            actual_device = "cpu"

    # Optimize CPU multi-threading for maximum inference speed
    if actual_device == "cpu":
        try:
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
        except Exception:
            pass

    cache_key = (model_id_or_path, actual_device)
    if _CACHED_VOXCPM_MODEL is not None and _CACHED_MODEL_KEY == cache_key:
        model = _CACHED_VOXCPM_MODEL
        if status_callback:
            status_callback(f"ប្រើប្រាស់ Model ក្នុង RAM ស្រាប់ មិនបាច់ផ្ទុក 5GB ឡើងវិញ ({actual_device.upper()})...")
    else:
        if status_callback:
            status_callback(f"កំពុងផ្ទុក Model VoxCPM2 ចូល Memory ({actual_device.upper()})...")

        model = VoxCPM.from_pretrained(
            model_id_or_path,
            load_denoiser=False,
            optimize=False,
            device=actual_device
        )
        _CACHED_VOXCPM_MODEL = model
        _CACHED_MODEL_KEY = cache_key

    if status_callback:
        status_callback("កំពុងសំយោគ និង Clone សំឡេង (Generating 48kHz Audio)...")

    if prompt_text:
        wav_tensor = model.generate(
            text=text,
            prompt_wav_path=prompt_audio_16k,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps
        )
    else:
        wav_tensor = model.generate(
            text=text,
            reference_wav_path=prompt_audio_16k,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps
        )

    native_sr = getattr(model.tts_model, "sample_rate", 48000)
    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(output_wav_path, wav_tensor, native_sr)
    except Exception:
        import time
        fallback_path = str(Path(output_wav_path).with_name(f"cloned_voice_{int(time.time() * 1000)}.wav"))
        sf.write(fallback_path, wav_tensor, native_sr)
        return fallback_path
    return output_wav_path


def fast_neural_voice_match(
    text: str,
    ref_audio_path: str,
    output_wav_path: str,
    status_callback = None
) -> str:
    """
    Instant smart voice cloning fallback:
    Analyzes the pitch and gender of the reference audio with librosa/ffmpeg,
    matches the exact pitch offset (+X Hz), and synthesizes with Edge-TTS in under 1 second!
    """
    import asyncio
    import librosa
    import numpy as np
    try:
        from tts_engine import generate_edge_tts_async
    except Exception:
        async def generate_edge_tts_async(text: str, voice: str, output_path: str, rate: str = "+0%", pitch: str = "+0Hz"):
            import edge_tts
            clean = text.strip() or "..."
            comm = edge_tts.Communicate(clean, voice, rate=rate, pitch=pitch)
            await comm.save(output_path)

    if status_callback:
        status_callback("កំពុងវិភាគសំឡេងគំរូ (Pitch & Timbre Matching)...")

    # Analyze reference audio pitch using librosa (first 4 seconds for high-speed analysis)
    try:
        y, sr = librosa.load(ref_audio_path, sr=16000, mono=True, duration=4.0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=16000
        )
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else []
        mean_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 160.0
    except Exception:
        mean_pitch = 160.0

    # Classify gender and compute pitch offset
    if mean_pitch > 165.0:
        voice_id = "km-KH-SreymomNeural"
        offset_hz = int(np.clip(mean_pitch - 210.0, -30.0, 30.0))
    else:
        voice_id = "km-KH-PisethNeural"
        offset_hz = int(np.clip(mean_pitch - 130.0, -30.0, 30.0))

    pitch_str = f"{offset_hz:+d}Hz"

    if status_callback:
        status_callback(f"កំពុងបង្កើតសំឡេងភាសាខ្មែរ (Pitch: {pitch_str})...")

    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    temp_mp3 = str(Path(output_wav_path).with_suffix(".mp3"))
    asyncio.run(generate_edge_tts_async(text, voice_id, temp_mp3, rate="+0%", pitch=pitch_str))

    # Convert to 48kHz WAV
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", temp_mp3,
        "-ar", "48000",
        output_wav_path
    ]
    silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(temp_mp3):
        try:
            os.remove(temp_mp3)
        except Exception:
            pass

    return output_wav_path


def mux_cloned_voice_to_video_fast(
    video_path: str,
    cloned_audio_path: str,
    output_video_path: str,
    ffmpeg_exe: str = FFMPEG_EXE
) -> str:
    """
    Merges cloned voice into video with -c:v copy (zero re-encoding, extremely fast).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"រកមិនឃើញវីដេអូដើម: {video_path}")
    if not os.path.exists(cloned_audio_path):
        raise FileNotFoundError(f"រកមិនឃើញសំឡេង Cloned: {cloned_audio_path}")

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-i", cloned_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_video_path
    ]

    res = silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg Muxing Error: {res.stderr}")
    return output_video_path


class VoiceCloneWorker(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    cloning_finished = pyqtSignal(str)  # output_audio_path
    row_finished = pyqtSignal(int, str)  # (row_index, line_output_path)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        text: str,
        ref_audio_path: str,
        output_wav_path: str,
        mode: str = "voxcpm",
        prompt_text: str = None,
        model_path: str = "openbmb/VoxCPM2",
        cfg_value: float = 2.0,
        timesteps: int = 10,
        device: str = "auto",
        api_provider: str = "elevenlabs",
        api_key: str = "",
        api_endpoint: str = "",
        voice_id: str = "",
        speed: float = 1.0,
        emotion: str = "normal",
        style_prompt: str = "",
        batch_items: list = None,
        colab_url: str = "",
        ref_audio_path_female: str = None,
        ref_audio_path_male: str = None,
        merge_mode: str = "continuous",
        trim_leading_silence: bool = True
    ):
        super().__init__()
        self.text = text
        self.ref_audio_path = ref_audio_path
        self.output_wav_path = output_wav_path
        self.mode = mode
        self.prompt_text = prompt_text
        self.model_path = model_path
        self.cfg_value = cfg_value
        self.timesteps = timesteps
        self.device = device
        self.api_provider = api_provider
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.voice_id = voice_id
        self.speed = speed
        self.emotion = emotion
        self.style_prompt = style_prompt
        self.batch_items = batch_items or []
        self.colab_url = colab_url
        self.ref_audio_path_female = ref_audio_path_female
        self.ref_audio_path_male = ref_audio_path_male
        self.merge_mode = merge_mode
        self.trim_leading_silence = trim_leading_silence

    def _merge_subtitle_audios(
        self,
        items: list,
        output_path: str,
        sample_rate: int = 48000,
        mode: str = "continuous",
        trim_leading: bool = True,
        pause_ms: int = 350
    ) -> str:
        import soundfile as sf
        import numpy as np
        import math
        from pathlib import Path

        valid_clips = []
        for it in items:
            p = getattr(it, "audio_path", "")
            if p and os.path.exists(p) and os.path.getsize(p) > 100:
                s_ms = getattr(it, "start_ms", 0)
                e_ms = getattr(it, "end_ms", s_ms + 1000)
                valid_clips.append((s_ms, e_ms, p))

        if not valid_clips:
            raise RuntimeError("គ្មានឯកសារសំឡេងបន្ទាត់ណាមួយត្រូវបានបង្កើតជោគជ័យឡើយ (No valid clips generated)")

        # Sort clips chronologically by start timestamp
        valid_clips.sort(key=lambda x: x[0])

        loaded = []
        for s_ms, e_ms, clip_path in valid_clips:
            try:
                data, sr = sf.read(clip_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != sample_rate:
                    try:
                        dur = len(data) / float(sr)
                        num_target = int(round(dur * sample_rate))
                        orig_times = np.linspace(0, dur, len(data), endpoint=False)
                        target_times = np.linspace(0, dur, num_target, endpoint=False)
                        data = np.interp(target_times, orig_times, data).astype(np.float32)
                    except Exception as _r_err:
                        print(f"Resample error {clip_path}: {_r_err}")
                loaded.append((s_ms, e_ms, data.astype(np.float32)))
            except Exception as e:
                print(f"Error loading clip {clip_path}: {e}")

        if not loaded:
            raise RuntimeError("មិនអាចអានឯកសារសំឡេងបន្ទាត់នីមួយៗបានទេ (Failed to load audio clips)")

        if mode == "continuous":
            # Sequence clips consecutively with a pleasant breath pause (e.g. 350ms)
            segments = []
            pause_samples = np.zeros(int((pause_ms / 1000.0) * sample_rate), dtype=np.float32)
            for i, (s_ms, e_ms, d) in enumerate(loaded):
                segments.append(d)
                if i < len(loaded) - 1:
                    segments.append(pause_samples)
            buffer = np.concatenate(segments)
        else:
            # Timeline video sync mode
            min_start_ms = loaded[0][0] if trim_leading else 0
            max_end_ms = max(e_ms for s_ms, e_ms, d in loaded)
            total_samples = int(((max_end_ms - min_start_ms + 2500) / 1000.0) * sample_rate)
            buffer = np.zeros(total_samples, dtype=np.float32)
            curr_sample_idx = 0

            for s_ms, e_ms, d in loaded:
                pos_ms = max(0, s_ms - min_start_ms)
                start_idx = max(curr_sample_idx, int((pos_ms / 1000.0) * sample_rate))
                end_idx = start_idx + len(d)
                if end_idx > len(buffer):
                    buffer = np.pad(buffer, (0, end_idx - len(buffer)))
                buffer[start_idx:end_idx] += d
                # Prevent overlapping dialogue: next clip cannot start before this one finishes
                curr_sample_idx = end_idx + int((80 / 1000.0) * sample_rate)

        # Normalize peak amplitude to 0.95 for clear, studio sound
        max_val = np.max(np.abs(buffer))
        if max_val > 0.001:
            buffer = buffer * (0.95 / max_val)

        # Ensure destination folder exists!
        dest_p = Path(output_path).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest_p), buffer, sample_rate)

        if not dest_p.exists() or dest_p.stat().st_size < 500:
            raise RuntimeError(f"បរាជ័យក្នុងការសរសេរឯកសារសំឡេង: {dest_p}")

        return str(dest_p)

    def _merge_subtitle_audios_timeline(self, items: list, output_path: str, sample_rate: int = 48000) -> str:
        return self._merge_subtitle_audios(items, output_path, sample_rate=sample_rate, mode="timeline", trim_leading=True)

    def _merge_consecutive_audio_files(self, file_paths: list, output_path: str, pause_ms: int = 220, sample_rate: int = 48000) -> str:
        """Merges multiple consecutive audio chunks with a natural human breath pause."""
        import soundfile as sf
        import numpy as np
        from pathlib import Path

        loaded = []
        for p in file_paths:
            if not p or not os.path.exists(p) or os.path.getsize(p) < 100:
                continue
            try:
                data, sr = sf.read(p)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != sample_rate:
                    dur = len(data) / float(sr)
                    num_target = int(round(dur * sample_rate))
                    orig_times = np.linspace(0, dur, len(data), endpoint=False)
                    target_times = np.linspace(0, dur, num_target, endpoint=False)
                    data = np.interp(target_times, orig_times, data).astype(np.float32)
                t_data = trim_audio_edges(data.astype(np.float32), sample_rate, threshold=0.005, pad_ms=30)
                if len(t_data) > 0:
                    loaded.append(t_data)
            except Exception as e:
                print(f"Error loading chunk audio {p}: {e}")

        if not loaded:
            raise RuntimeError("គ្មានឯកសារសំឡេងឃ្លាណាមួយត្រឹមត្រូវសម្រាប់ផ្គុំឡើយ")

        segments = []
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

        dest_p = Path(output_path).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest_p), buffer, sample_rate)
        return str(dest_p)

    def _synthesize_single_line(self, line_text: str, effective_ref: str, out_path: str) -> str:
        """Synthesizes line with auto-chunking if line is long (>85 chars) to prevent AI language drift."""
        from api_voice_cloner import split_khmer_chunks
        if len(line_text.strip()) > 90:
            sub_chunks = split_khmer_chunks(line_text, max_chars=90)
            if len(sub_chunks) > 1:
                import uuid
                sub_audios = []
                for sub_i, sub_t in enumerate(sub_chunks):
                    sub_out = str(CACHE_DIR / f"subrow_{uuid.uuid4().hex[:8]}.wav")
                    sub_res = self._synthesize_raw_line(sub_t, effective_ref, sub_out)
                    if sub_res and os.path.exists(sub_res) and os.path.getsize(sub_res) > 300:
                        sub_audios.append(sub_res)
                if sub_audios:
                    merged = self._merge_consecutive_audio_files(sub_audios, out_path, pause_ms=140)
                    for sa in sub_audios:
                        try:
                            os.unlink(sa)
                        except Exception:
                            pass
                    return merged
        return self._synthesize_raw_line(line_text, effective_ref, out_path)

    def _synthesize_raw_line(self, line_text: str, effective_ref: str, out_path: str) -> str:
        from api_voice_cloner import apply_audio_emotion_and_speed
        if self.mode == "colab_voxcpm":
            from api_voice_cloner import clone_voice_colab_api
            effective_cfg = self.cfg_value
            if self.emotion == "storyteller":
                effective_cfg = 1.9
            elif self.emotion == "news_anchor":
                effective_cfg = 2.1
            elif self.emotion == "angry":
                effective_cfg = 2.6
            elif self.emotion == "fast_action":
                effective_cfg = 2.0
            elif self.emotion in ["sad_gentle", "gentle", "sad"]:
                effective_cfg = 1.6
            elif self.emotion == "fear":
                effective_cfg = 2.2

            # Only send prompt_text if user explicitly typed a reference audio transcript
            real_ref_transcript = self.prompt_text.strip() if (self.prompt_text and self.prompt_text.strip()) else None

            out = clone_voice_colab_api(
                text=line_text,
                colab_url=self.colab_url,
                output_path=out_path,
                ref_audio_path=effective_ref,
                prompt_text=real_ref_transcript,
                cfg_value=effective_cfg,
                timesteps=self.timesteps,
                speed=self.speed,
                emotion=self.emotion
            )
            return out
        elif self.mode in ["edge_tts", "khmer_ai"]:
            voice = self.voice_id or "km-KH-PisethNeural"
            import asyncio, edge_tts, time
            clean = line_text.strip() or "..."
            temp_mp3 = str(Path(out_path).with_suffix(".mp3"))

            # Robust auto-retry engine for unstable / jittery internet connections (up to 4 attempts)
            last_err = None
            for attempt in range(4):
                try:
                    if attempt > 0:
                        self.status_updated.emit(f"កំពុងព្យាយាមតភ្ជាប់អ៊ីនធឺណិតឡើងវិញ (លើកទី {attempt + 1}/4)...")
                    async def _run_tts():
                        comm = edge_tts.Communicate(clean, voice)
                        await comm.save(temp_mp3)
                    asyncio.run(_run_tts())
                    if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 300:
                        last_err = None
                        break
                except Exception as ex:
                    last_err = ex
                    time.sleep(1.2 * (attempt + 1))

            if last_err and (not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) <= 300):
                raise RuntimeError(
                    f"🌐 បញ្ហាដាច់ការតភ្ជាប់អ៊ីនធឺណិត: ប្រព័ន្ធបានព្យាយាមភ្ជាប់ឡើងវិញ ៤ ដង តែបរាជ័យដោយសារអ៊ីនធឺណិតរអាក់រអួលខ្លាំង ឬដាច់ ({str(last_err)})"
                )

            cmd = [
                FFMPEG_EXE, "-y",
                "-i", temp_mp3,
                "-ar", "48000",
                out_path
            ]
            silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
            return apply_audio_emotion_and_speed(out_path, out_path, speed=self.speed, emotion=self.emotion)
        elif self.mode == "cloud_api":
            if self.api_provider == "elevenlabs":
                from api_voice_cloner import clone_voice_elevenlabs_api
                out = clone_voice_elevenlabs_api(
                    text=line_text,
                    api_key=self.api_key,
                    output_path=out_path,
                    ref_audio_path=effective_ref,
                    voice_id=self.voice_id,
                    emotion=self.emotion,
                    speed=self.speed
                )
                return apply_audio_emotion_and_speed(out, out, speed=self.speed, emotion=self.emotion)
            elif self.api_provider == "kiritts":
                from api_voice_cloner import synthesize_kiritts_api
                out = synthesize_kiritts_api(
                    text=line_text,
                    api_key=self.api_key,
                    voice_id=self.voice_id or "Maly",
                    ref_audio_path=effective_ref if (effective_ref and os.path.exists(effective_ref)) else None,
                    output_path=out_path,
                    endpoint_url=self.api_endpoint or "https://api.kiritts.com/v1/audio/speech",
                    speed=self.speed,
                    emotion=self.emotion,
                    style_prompt=self.style_prompt
                )
                return out
            else:
                from api_voice_cloner import clone_voice_custom_cloud_api
                out = clone_voice_custom_cloud_api(
                    text=line_text,
                    ref_audio_path=effective_ref,
                    api_endpoint=self.api_endpoint,
                    api_key=self.api_key,
                    output_path=out_path
                )
                return apply_audio_emotion_and_speed(out, out, speed=self.speed, emotion=self.emotion)
        elif self.mode == "fast_neural":
            out = fast_neural_voice_match(
                text=line_text,
                ref_audio_path=effective_ref,
                output_wav_path=out_path,
                status_callback=None
            )
            return apply_audio_emotion_and_speed(out, out, speed=self.speed, emotion=self.emotion)
        else: # voxcpm
            effective_cfg = self.cfg_value
            if self.emotion == "storyteller":
                effective_cfg = 1.9
            elif self.emotion == "news_anchor":
                effective_cfg = 2.1
            elif self.emotion == "angry":
                effective_cfg = 2.6
            elif self.emotion == "fast_action":
                effective_cfg = 2.0
            elif self.emotion in ["sad_gentle", "gentle", "sad"]:
                effective_cfg = 1.6
            elif self.emotion == "fear":
                effective_cfg = 2.2

            # Only send prompt_text if user explicitly typed a reference audio transcript
            real_ref_transcript = self.prompt_text.strip() if (self.prompt_text and self.prompt_text.strip()) else None

            out = clone_voice_voxcpm2_core(
                text=line_text,
                prompt_audio_16k=effective_ref,
                output_wav_path=out_path,
                prompt_text=real_ref_transcript,
                model_id_or_path=self.model_path,
                cfg_value=effective_cfg,
                inference_timesteps=self.timesteps,
                device=self.device,
                status_callback=None
            )
            return apply_audio_emotion_and_speed(out, out, speed=self.speed, emotion=self.emotion)

    def run(self):
        try:
            import time
            import uuid
            # CACHE_DIR globally defined
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # 1. Preprocess reference audios for male and female
            temp_16k_m = None
            temp_16k_f = None
            effective_ref_m = self.ref_audio_path_male or self.ref_audio_path
            effective_ref_f = self.ref_audio_path_female or self.ref_audio_path

            if self.mode in ["fast_neural", "voxcpm", "colab_voxcpm"]:
                if effective_ref_m and os.path.exists(effective_ref_m):
                    self.status_updated.emit("កំពុងត្រៀមសំឡេងគំរូប្រុស 16kHz...")
                    self.progress_updated.emit(3)
                    temp_16k_m = str(CACHE_DIR / f"temp_vox_ref_m_{int(time.time() * 1000)}.wav")
                    preprocess_reference_audio_16k(effective_ref_m, temp_16k_m)
                    effective_ref_m = temp_16k_m

                if effective_ref_f and os.path.exists(effective_ref_f) and effective_ref_f != self.ref_audio_path_male:
                    self.status_updated.emit("កំពុងត្រៀមសំឡេងគំរូស្រី 16kHz...")
                    self.progress_updated.emit(5)
                    temp_16k_f = str(CACHE_DIR / f"temp_vox_ref_f_{int(time.time() * 1000)}.wav")
                    preprocess_reference_audio_16k(effective_ref_f, temp_16k_f)
                    effective_ref_f = temp_16k_f

            # 2. Batch Mode: Clone all subtitle rows line-by-line using gender-specific reference
            if self.batch_items and len(self.batch_items) > 0:
                total = len(self.batch_items)
                for idx, (item_id, item_text, s_obj) in enumerate(self.batch_items):
                    pct = int((idx / total) * 85) + 5
                    self.progress_updated.emit(pct)

                    s_gen = getattr(s_obj, "gender", "male").lower() if s_obj else "male"
                    if s_gen == "female" and self.ref_audio_path_female and effective_ref_f:
                        line_ref = effective_ref_f
                        gen_lbl = "👧 ស្រី"
                    else:
                        line_ref = effective_ref_m or self.ref_audio_path or effective_ref_f
                        gen_lbl = "👦 ប្រុស"

                    short_t = (item_text[:24] + "...") if len(item_text) > 24 else item_text
                    self.status_updated.emit(f"កំពុង Clone [{idx + 1}/{total} - {gen_lbl}]: {short_t}")

                    # Strictly enforce 1:1 file mapping per row: purge previous files for this item_id
                    for old_f in CACHE_DIR.glob(f"cloned_line_{item_id}_*.wav"):
                        try:
                            old_f.unlink(missing_ok=True)
                        except Exception:
                            pass
                    for old_f in CACHE_DIR.glob(f"voice_row_{item_id}_*.mp3"):
                        try:
                            old_f.unlink(missing_ok=True)
                        except Exception:
                            pass

                    line_out = str(CACHE_DIR / f"cloned_line_{item_id}.wav")
                    res_path = self._synthesize_single_line(item_text, line_ref, line_out)
                    if s_obj:
                        s_obj.audio_path = res_path
                        s_obj.status = "Cloned"
                    self.row_finished.emit(idx, res_path)

                for t_file in [temp_16k_m, temp_16k_f]:
                    if t_file and os.path.exists(t_file):
                        try:
                            os.unlink(t_file)
                        except Exception:
                            pass

                self.status_updated.emit("កំពុងផ្គុំសំឡេងគ្រប់បន្ទាត់បញ្ចូលគ្នា (Building Soundtrack)...")
                self.progress_updated.emit(92)
                sub_list = [it[2] for it in self.batch_items if it[2]]
                try:
                    full_audio = self._merge_subtitle_audios(
                        sub_list,
                        self.output_wav_path,
                        sample_rate=48000,
                        mode=getattr(self, "merge_mode", "continuous"),
                        trim_leading=getattr(self, "trim_leading_silence", True)
                    )
                except Exception as e:
                    self.error_occurred.emit(f"មានបញ្ហាក្នុងការផ្គុំសំឡេង: {str(e)}")
                    return

                if not full_audio or not os.path.exists(full_audio) or os.path.getsize(full_audio) < 500:
                    self.error_occurred.emit("ឯកសារសំឡេងដែលបង្កើតរួចមិនមានទិន្នន័យ (Empty output audio file)។")
                    return

                self.progress_updated.emit(100)
                self.status_updated.emit(f"Clone សំឡេងបានជោគជ័យ 100% (សរុប {total} ឃ្លា)!")
                self.cloning_finished.emit(full_audio)
                return

            # 3. Single Mode: Clone active text with Smart Auto-Chunking (anti-hallucination)
            from api_voice_cloner import split_khmer_chunks
            
            # Resolve effective reference audio for single mode
            effective_ref = self.ref_audio_path
            if not effective_ref or not os.path.exists(effective_ref):
                effective_ref = effective_ref_m or effective_ref_f

            if effective_ref == self.ref_audio_path_male and temp_16k_m and os.path.exists(temp_16k_m):
                effective_ref = temp_16k_m
            elif effective_ref == self.ref_audio_path_female and temp_16k_f and os.path.exists(temp_16k_f):
                effective_ref = temp_16k_f
            elif temp_16k_m and os.path.exists(temp_16k_m):
                effective_ref = temp_16k_m

            chunks = split_khmer_chunks(self.text, max_chars=90)
            if len(chunks) <= 1:
                self.progress_updated.emit(40)
                self.status_updated.emit("កំពុងដំណើរការ Clone សំឡេង...")
                out_audio = self._synthesize_single_line(self.text, effective_ref, self.output_wav_path)
            else:
                chunk_audios = []
                total_c = len(chunks)
                for c_i, c_text in enumerate(chunks):
                    pct = int(10 + (c_i / total_c) * 80)
                    self.progress_updated.emit(pct)
                    short_disp = (c_text[:24] + "...") if len(c_text) > 24 else c_text
                    self.status_updated.emit(f"កំពុង Clone ឃ្លាទី [{c_i + 1}/{total_c}]: {short_disp}")
                    
                    c_out = str(CACHE_DIR / f"single_chunk_{uuid.uuid4().hex[:8]}.wav")
                    c_res = self._synthesize_raw_line(c_text, effective_ref, c_out)
                    if c_res and os.path.exists(c_res) and os.path.getsize(c_res) > 300:
                        chunk_audios.append(c_res)

                if not chunk_audios:
                    raise RuntimeError("មិនអាចបង្កើតសំឡេងឃ្លាណាមួយបានឡើយ (Failed to synthesize chunks)")

                self.status_updated.emit("កំពុងផ្គុំសំឡេងគ្រប់ឃ្លាបញ្ចូលគ្នា (Merging sentences)...")
                self.progress_updated.emit(94)
                
                # Merge clips with natural breath pause
                out_audio = self._merge_consecutive_audio_files(chunk_audios, self.output_wav_path, pause_ms=140)
                
                # Cleanup temp chunks
                for ca in chunk_audios:
                    try:
                        os.unlink(ca)
                    except Exception:
                        pass

            for t_file in [temp_16k_m, temp_16k_f]:
                if t_file and os.path.exists(t_file):
                    try:
                        os.unlink(t_file)
                    except Exception:
                        pass

            self.progress_updated.emit(100)
            self.status_updated.emit("Clone សំឡេងបានជោគជ័យ 100% (Studio 48kHz)!")
            self.cloning_finished.emit(out_audio)

        except Exception as e:
            self.error_occurred.emit(str(e))


class VoiceCloneDialog(QDialog):
    """
    Modern Voice Cloning Studio Dialog powered by VoxCPM2 & Neural Pitch Match.
    """
    cloned_audio_ready = pyqtSignal(str)

    def __init__(self, parent=None, current_video_path="", subtitles=None, active_text="", is_standalone=None):
        super().__init__(parent)
        self.is_standalone = is_standalone if is_standalone is not None else (subtitles is None and not current_video_path and parent is None)
        self.setWindowTitle("🎙️ VoxCPM2 Voice Cloning Studio - Cambo Dubber")
        app_dir = Path(__file__).resolve().parent
        ico_file = app_dir / "assets" / "voice_clone_logo.ico"
        if not ico_file.exists():
            ico_file = app_dir / "assets" / "voice_clone_logo.png"
        if ico_file.exists():
            from PyQt6.QtGui import QIcon
            self.setWindowIcon(QIcon(str(ico_file)))
        self.resize(1020, 840)
        # បើកប៊ូតុង Maximize និង Minimize លើ Dialog ដើម្បីឱ្យអ្នកប្រើអាចពង្រីកពេញ Screen
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("""
            QDialog {
                background-color: #0b0f19;
                color: #f8fafc;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: bold;
                color: #10b981;
                background-color: #0f172a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #10b981;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QLineEdit, QTextEdit {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #7c3aed;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #64748b;
            }
            QPushButton.btn-primary {
                background-color: #7c3aed;
                border: 1px solid #6d28d9;
                font-size: 12px;
                padding: 8px 16px;
            }
            QPushButton.btn-primary:hover {
                background-color: #8b5cf6;
                border-color: #a78bfa;
            }
            QPushButton.btn-success {
                background-color: #059669;
                border: 1px solid #047857;
            }
            QPushButton.btn-success:hover {
                background-color: #10b981;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 5px;
                background-color: #1e293b;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #7c3aed;
                border-radius: 4px;
            }
            /* Right-Click Context Menu */
            QMenu {
                background-color: #0f172a;
                color: #38bdf8;
                border: 1px solid #0284c7;
                border-radius: 6px;
                padding: 5px;
                font-size: 11px;
                font-weight: 500;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 24px 6px 14px;
                border-radius: 4px;
                color: #38bdf8;
            }
            QMenu::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #64748b;
                background-color: transparent;
            }
            QMenu::separator {
                height: 1px;
                background-color: #334155;
                margin: 4px 6px;
            }
        """)

        self.current_video_path = current_video_path
        self.subtitles = subtitles or []
        self.active_text = active_text
        self.output_cloned_wav = ""
        self.is_batch_mode = False

        self.player_ref = QMediaPlayer()
        self.audio_ref = QAudioOutput()
        self.player_ref.setAudioOutput(self.audio_ref)

        self.player_res = QMediaPlayer()
        self.audio_res = QAudioOutput()
        self.player_res.setAudioOutput(self.audio_res)
        self.player_res.playbackStateChanged.connect(self._on_player_res_playback_changed)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header Title
        header_layout = QHBoxLayout()
        lbl_title = QLabel("🎙️ VoxCPM2 Voice Cloning Studio (48kHz)")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #a78bfa;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()

        if self.is_standalone:
            lbl_badge = QLabel("☁️ Google Colab GPU Edition (48kHz)")
            lbl_badge.setStyleSheet("background-color: #064e3b; color: #6ee7b7; padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: bold;")
        else:
            is_ready = is_voxcpm_ready()
            if is_ready:
                lbl_badge = QLabel("✅ VoxCPM2 Core Ready")
                lbl_badge.setStyleSheet("background-color: #064e3b; color: #6ee7b7; padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: bold;")
            else:
                lbl_badge = QLabel("⚠️ Core Installing...")
                lbl_badge.setStyleSheet("background-color: #78350f; color: #fde68a; padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: bold;")
        header_layout.addWidget(lbl_badge)
        main_layout.addLayout(header_layout)

        self.scroll = QScrollArea()
        scroll = self.scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(12)

        # --- SECTION 0: Mode Selection ---
        self.grp_mode = QGroupBox("ជ្រើសរើសទម្រង់ Clone សំឡេង (Voice Cloning Mode)")
        v_mode = QVBoxLayout(self.grp_mode)

        if self.is_standalone:
            lbl_mode_info = QLabel("🚀 វិធីសាស្ត្រដំណើរការ: Google Colab GPU Server (VoxCPM2 Foundation Model 48kHz - ឥតគិតថ្លៃ & លឿន)")
            lbl_mode_info.setStyleSheet("font-weight: bold; color: #10b981; font-size: 12px; padding: 4px;")
            v_mode.addWidget(lbl_mode_info)
            self.combo_clone_mode = QComboBox()
            self.combo_clone_mode.addItem("Google Colab GPU")
            self.combo_clone_mode.setVisible(False)
        else:
            h_mode_box = QHBoxLayout()
            lbl_mode = QLabel("វិធីសាស្ត្រ Clone:")
            lbl_mode.setStyleSheet("font-weight: bold; color: #10b981; font-size: 11px;")
            self.combo_clone_mode = QComboBox()
            self.combo_clone_mode.addItems([
                "💻 Local Offline AI (VoxCPM2 - ក្នុងកុំព្យូទ័រផ្ទាល់ 48kHz)",
                "🚀 Google Colab GPU Server (VoxCPM2 + Ngrok - ឥតគិតថ្លៃ & លឿន)",
                "🌐 Cloud API Clone (ElevenLabs / Website API - មិនស៊ីកម្លាំងម៉ាស៊ីន)",
                "⚡ Instant Match (Microsoft Edge-TTS - ឥតគិតថ្លៃ & លឿន)"
            ])
            self.combo_clone_mode.setStyleSheet("font-weight: bold; color: #10b981; padding: 4px;")
            self.combo_clone_mode.currentIndexChanged.connect(self._on_mode_changed)
            h_mode_box.addWidget(lbl_mode)
            h_mode_box.addWidget(self.combo_clone_mode, 1)
            v_mode.addLayout(h_mode_box)
        layout.addWidget(self.grp_mode)

        # --- SECTION 1: Reference Audio (Supports Dual Male & Female Reference Samples) ---
        grp_ref = QGroupBox("១. សំឡេងគំរូដើម (Reference Audio Sample - 3 ដល់ 10 វិនាទី)")
        v_ref = QVBoxLayout(grp_ref)
        v_ref.setSpacing(6)

        # 1. Male / Primary Reference Row
        h_ref_male = QHBoxLayout()
        self.lbl_m = QLabel("🎙️ សំឡេងគំរូដើម:" if self.is_standalone else "👦 សំឡេងប្រុស:")
        self.lbl_m.setFixedWidth(130 if self.is_standalone else 90)
        self.lbl_m.setStyleSheet("font-weight: bold; color: #93c5fa; font-size: 11px;")
        self.txt_ref_male = QLineEdit()
        self.txt_ref_male.setPlaceholderText("ជ្រើសរើសសំឡេងគំរូ ៣ ដល់ ១០ វិនាទី (.wav / .mp3)..." if self.is_standalone else "ជ្រើសរើសសំឡេងគំរូមនុស្សប្រុស (.wav / .mp3)...")
        self.btn_browse_male = QPushButton("📁 រើសសំឡេងគំរូ" if self.is_standalone else "📁 រើសសំឡេងប្រុស")
        self.btn_browse_male.setStyleSheet("background-color: #1e3a5f; color: #93c5fa; border: 1px solid #2563eb;")
        self.btn_browse_male.clicked.connect(lambda: self.on_browse_ref_audio("male"))

        self.btn_ext_male = QPushButton("🎬 ទាញពីវីដេអូ")
        self.btn_ext_male.clicked.connect(lambda: self.on_extract_from_video("male"))
        if not self.current_video_path or not os.path.exists(self.current_video_path) or self.is_standalone:
            self.btn_ext_male.setVisible(False)

        self.btn_play_male = QPushButton("▶ ស្តាប់")
        self.btn_play_male.clicked.connect(lambda: self.on_toggle_play_ref("male"))

        h_ref_male.addWidget(self.lbl_m)
        h_ref_male.addWidget(self.txt_ref_male, 1)
        h_ref_male.addWidget(self.btn_browse_male)
        h_ref_male.addWidget(self.btn_ext_male)
        h_ref_male.addWidget(self.btn_play_male)
        v_ref.addLayout(h_ref_male)

        # 2. Female Reference Row
        self.row_female = QWidget()
        h_ref_female = QHBoxLayout(self.row_female)
        h_ref_female.setContentsMargins(0, 0, 0, 0)
        self.lbl_f = QLabel("👧 សំឡេងស្រី:")
        self.lbl_f.setFixedWidth(90)
        self.lbl_f.setStyleSheet("font-weight: bold; color: #f472b6; font-size: 11px;")
        self.txt_ref_female = QLineEdit()
        self.txt_ref_female.setPlaceholderText("ជ្រើសរើសសំឡេងគំរូមនុស្សស្រី (.wav / .mp3 - ស្រេចចិត្ត)...")
        self.btn_browse_female = QPushButton("📁 រើសសំឡេងស្រី")
        self.btn_browse_female.setStyleSheet("background-color: #4a152e; color: #f472b6; border: 1px solid #db2777;")
        self.btn_browse_female.clicked.connect(lambda: self.on_browse_ref_audio("female"))

        self.btn_ext_female = QPushButton("🎬 ទាញពីវីដេអូ")
        self.btn_ext_female.clicked.connect(lambda: self.on_extract_from_video("female"))
        if not self.current_video_path or not os.path.exists(self.current_video_path) or self.is_standalone:
            self.btn_ext_female.setVisible(False)

        self.btn_play_female = QPushButton("▶ ស្តាប់")
        self.btn_play_female.clicked.connect(lambda: self.on_toggle_play_ref("female"))

        h_ref_female.addWidget(self.lbl_f)
        h_ref_female.addWidget(self.txt_ref_female, 1)
        h_ref_female.addWidget(self.btn_browse_female)
        h_ref_female.addWidget(self.btn_ext_female)
        h_ref_female.addWidget(self.btn_play_female)
        v_ref.addWidget(self.row_female)

        # Alias for legacy compatibility
        self.txt_ref_path = self.txt_ref_male
        self.btn_play_ref = self.btn_play_male
        self.btn_extract_video = self.btn_ext_male

        self.lbl_p_text = QLabel("អត្ថបទដើមនៃសំឡេងគំរូ (Prompt Text Transcript - ស្រេចចិត្ត/Optional):")
        self.txt_prompt_text = QLineEdit()
        self.txt_prompt_text.setPlaceholderText("សរសេរពាក្យដែលសំឡេងគំរូនិយាយ (ប្រសិនបើដឹង ជួយឱ្យសំឡេងដូច 100%)...")
        v_ref.addWidget(self.lbl_p_text)
        v_ref.addWidget(self.txt_prompt_text)

        if self.is_standalone:
            self.row_female.setVisible(False)
            self.lbl_p_text.setVisible(False)
            self.txt_prompt_text.setVisible(False)

        layout.addWidget(grp_ref)

        # --- SECTION 2: Text to Synthesize (Toggle between Single Line & Subtitle Table) ---
        grp_text = QGroupBox("២. អត្ថបទដែលត្រូវនិយាយ (Text to Synthesize)")
        v_text = QVBoxLayout(grp_text)
        v_text.setSpacing(6)

        sub_count = len(self.subtitles) if self.subtitles else 0
        self.radio_single = QRadioButton("អត្ថបទបន្ទាត់នេះ (Active Text Line)")
        self.radio_batch = QRadioButton(f"បកប្រែគ្រប់បន្ទាត់ក្នុងតារាង Subtitles ({sub_count} ឃ្លា - Dub All Rows)")
        self.radio_single.setChecked(True)

        h_radios = QHBoxLayout()
        h_radios.addWidget(self.radio_single)
        h_radios.addWidget(self.radio_batch)
        h_radios.addSpacing(12)

        # ---------------------------------------------------------------------
        # ប្រអប់ខាងឆ្វេងដៃ: Speed (ល្បឿនក្នុងការអាន / ល្បឿនក្លូន យឺត ឬលឿន)
        # ---------------------------------------------------------------------
        self.lbl_spd = QLabel("⏱️ Speed:")
        self.lbl_spd.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.combo_speed = QComboBox()
        self.combo_speed.addItem("0.80x (យឺតខ្លាំង)", 0.80)
        self.combo_speed.addItem("0.85x (យឺតល្មម)", 0.85)
        self.combo_speed.addItem("0.90x (យឺតបន្តិច)", 0.90)
        self.combo_speed.addItem("1.00x (ធម្មតា)", 1.00)
        self.combo_speed.addItem("1.10x (លឿនបន្តិច)", 1.10)
        self.combo_speed.addItem("1.20x (លឿន)", 1.20)
        self.combo_speed.addItem("1.30x (លឿនរហ័ស)", 1.30)
        self.combo_speed.addItem("1.50x (លឿនខ្លាំង)", 1.50)
        self.combo_speed.setCurrentIndex(3)  # Default: 1.00x
        self.combo_speed.setStyleSheet("""
            QComboBox {
                font-weight: bold;
                font-size: 11px;
                color: #38bdf8;
                background-color: #1e293b;
                border: 1px solid #3b82f6;
                border-radius: 5px;
                padding: 2px 8px;
                min-height: 24px;
            }
            QComboBox:hover {
                border: 1px solid #60a5fa;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f1f5f9;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
        """)

        h_radios.addWidget(self.lbl_spd)
        h_radios.addWidget(self.combo_speed)
        h_radios.addSpacing(10)

        # ---------------------------------------------------------------------
        # ប្រអប់ខាងស្តាំដៃ: Option / បែបបទអាន (១. អានព័ត៌មាន, ២. អានរឿងនិទាន, ៣. ផ្សេងៗ)
        # ---------------------------------------------------------------------
        self.lbl_emo = QLabel("🎙️ Option:")
        self.lbl_emo.setStyleSheet("color: #a78bfa; font-weight: bold; font-size: 11px;")
        self.combo_emotion = QComboBox()
        self.combo_emotion.addItem("📰 អានព័ត៌មាន (News Reading)", "news_anchor")
        self.combo_emotion.addItem("📖 អានរឿងនិទាន (Storyteller)", "storyteller")
        self.combo_emotion.addItem("🎭 ផ្សេងៗ (ធម្មតា / General)", "normal")
        self.combo_emotion.addItem("😡 ផ្សេងៗ (ខឹង / ម៉ឺងម៉ាត់)", "angry")
        self.combo_emotion.addItem("😢 ផ្សេងៗ (កម្សត់ / ស្រទន់)", "sad_gentle")
        self.combo_emotion.addItem("😨 ផ្សេងៗ (ភ័យខ្លាច)", "fear")
        self.combo_emotion.addItem("⚡ ផ្សេងៗ (និយាយលឿន / Action)", "fast_action")
        self.combo_emotion.setCurrentIndex(0)
        self.combo_emotion.setStyleSheet("""
            QComboBox {
                font-weight: bold;
                font-size: 11px;
                color: #a78bfa;
                background-color: #1e293b;
                border: 1px solid #8b5cf6;
                border-radius: 5px;
                padding: 2px 8px;
                min-height: 24px;
            }
            QComboBox:hover {
                border: 1px solid #c084fc;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f1f5f9;
                selection-background-color: #7c3aed;
                selection-color: #ffffff;
            }
        """)

        h_radios.addWidget(self.lbl_emo)
        h_radios.addWidget(self.combo_emotion)

        # ---------------------------------------------------------------------
        # ឈ្មោះអ្នកអាន (Default System Sounds for News Reading)
        # ---------------------------------------------------------------------
        self.combo_news_reader = QComboBox()
        self.combo_news_reader.addItem("👤 ឈ្មោះអ្នកអាន...", "")
        self.combo_news_reader.addItem("🎙️ ១. បែបបទយកការណ៍ (ប្រុស)", "news_reader_report_male.wav")
        self.combo_news_reader.addItem("🎙️ ២. Creator (ប្រុស)", "news_reader_creator_male.wav")
        self.combo_news_reader.addItem("🎙️ ៣. ទទក / TVK (ស្រី)", "news_reader_tvk_female.wav")
        self.combo_news_reader.setToolTip("ជ្រើសរើសសំឡេងគំរូអ្នកអានព័ត៌មាន System Default (មាន ៣ សំឡេងក្នុងប្រព័ន្ធ)")
        self.combo_news_reader.setStyleSheet("""
            QComboBox {
                font-weight: bold;
                font-size: 11px;
                color: #38bdf8;
                background-color: #1e293b;
                border: 1.5px solid #0284c7;
                border-radius: 5px;
                padding: 2px 8px;
                min-height: 24px;
            }
            QComboBox:hover {
                border: 1.5px solid #38bdf8;
            }
            QComboBox:disabled {
                color: #64748b;
                background-color: #0f172a;
                border: 1px solid #334155;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f1f5f9;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
            }
        """)
        self.combo_news_reader.currentIndexChanged.connect(self._on_news_reader_selected)
        self.combo_emotion.currentIndexChanged.connect(self._on_emotion_changed)

        h_radios.addWidget(self.combo_news_reader)

        # Divider line
        self.line_tab_div = QFrame()
        self.line_tab_div.setFrameShape(QFrame.Shape.VLine)
        self.line_tab_div.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_tab_div.setStyleSheet("color: #475569; margin: 0 4px;")

        # Segmented Tab Buttons (Matching user's red vertical line request)
        self.btn_tab_plain = QPushButton("📝 អត្ថបទធម្មតា (Plain Text)")
        self.btn_tab_plain.setCheckable(True)
        self.btn_tab_plain.setChecked(True)

        self.btn_tab_srt = QPushButton("🎬 Subtitle (only SRT format)")
        self.btn_tab_srt.setCheckable(True)

        self.btn_text_mode_group = QButtonGroup(self)
        self.btn_text_mode_group.addButton(self.btn_tab_plain, 0)
        self.btn_text_mode_group.addButton(self.btn_tab_srt, 1)

        tab_btn_style = """
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f1f5f9;
            }
            QPushButton:checked {
                background-color: #7c3aed;
                color: #ffffff;
                border: 1.5px solid #a78bfa;
            }
        """
        self.btn_tab_plain.setStyleSheet(tab_btn_style)
        self.btn_tab_srt.setStyleSheet(tab_btn_style)

        h_radios.addWidget(self.line_tab_div)
        h_radios.addWidget(self.btn_tab_plain)
        h_radios.addWidget(self.btn_tab_srt)
        h_radios.addStretch()

        self.lbl_text_counter = QLabel("អត្ថបទបន្ទាត់បច្ចុប្បន្ន")
        self.lbl_text_counter.setStyleSheet("color: #10b981; font-size: 11px; font-weight: bold;")
        h_radios.addWidget(self.lbl_text_counter)
        v_text.addLayout(h_radios)

        # Stack for Plain Text vs SRT Mode
        self.stack_text_mode = QStackedWidget()
        self.stack_text_mode.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Page 0: Single Plain Text Edit Area
        self.edit_text = QTextEdit()
        self.edit_text.setPlaceholderText("បញ្ចូលអត្ថបទដែលចង់ឱ្យសំឡេង Cloned និយាយនៅទីនេះ...")
        self.edit_text.setMinimumHeight(180)
        self.edit_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if self.active_text:
            self.edit_text.setPlainText(self.active_text)
        self.stack_text_mode.addWidget(self.edit_text)

        # Page 1: SRT Format Subtitle Area (Matching image 2)
        self.widget_srt = QWidget()
        v_srt = QVBoxLayout(self.widget_srt)
        v_srt.setContentsMargins(0, 0, 0, 0)
        v_srt.setSpacing(4)

        # Top Bar
        h_srt_bar = QHBoxLayout()
        self.btn_upload_srt = QPushButton("📁 ផ្ទុកហ្វាល់ SRT (Upload)")
        self.btn_upload_srt.setStyleSheet("background-color: #0369a1; color: white; font-weight: bold; font-size: 11px; padding: 4px 10px; border-radius: 5px;")
        self.btn_upload_srt.clicked.connect(self._on_upload_srt_file)

        self.lbl_srt_info = QLabel("Convert Subtitle to Speech (only SRT format)")
        self.lbl_srt_info.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")

        self.combo_srt_dub_mode = QComboBox()
        self.combo_srt_dub_mode.addItem("🗣️ អានតៗគ្នា (Continuous Speech)", "continuous")
        self.combo_srt_dub_mode.addItem("🎬 តាមម៉ោងវីដេអូ (Timeline Video Sync)", "timeline")
        self.combo_srt_dub_mode.setToolTip("ជ្រើសរើសរបៀបផ្គុំសំឡេង៖ 'អានតៗគ្នា' (គ្មានចន្លោះស្ងាត់ & មិនជាន់គ្នា) ឬ 'តាមម៉ោងវីដេអូ' (រក្សាទុកចន្លោះម៉ោងតាម SRT)")
        self.combo_srt_dub_mode.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #38bdf8;
                font-weight: bold;
                border: 1px solid #0284c7;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 10px;
            }
        """)

        self.chk_srt_trim_silence = QCheckBox("✂️ ចាប់ផ្តើមភ្លាមៗ (Trim Leading Silence)")
        self.chk_srt_trim_silence.setChecked(True)
        self.chk_srt_trim_silence.setToolTip("កាត់ចន្លោះស្ងាត់ចោលនៅដើមឃ្លាដំបូង ដើម្បីឱ្យសំឡេងនិយាយចាប់ផ្តើមភ្លាមៗ (មិនបាច់រង់ចាំចន្លោះស្ងាត់)")
        self.chk_srt_trim_silence.setStyleSheet("color: #34d399; font-size: 10px; font-weight: bold;")

        self.lbl_srt_dur = QLabel("⏱️ 00:00:00 / 50 នាទី")
        self.lbl_srt_dur.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")

        h_srt_bar.addWidget(self.btn_upload_srt)
        h_srt_bar.addWidget(self.lbl_srt_info)
        h_srt_bar.addWidget(self.combo_srt_dub_mode)
        h_srt_bar.addWidget(self.chk_srt_trim_silence)
        h_srt_bar.addStretch()
        h_srt_bar.addWidget(self.lbl_srt_dur)
        v_srt.addLayout(h_srt_bar)

        # SRT Text Edit
        self.edit_srt = QTextEdit()
        self.edit_srt.setAcceptRichText(False)
        self.edit_srt.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', 'Khmer OS Battambang', monospace;
                font-size: 12px;
                color: #e2e8f0;
                background-color: #1e293b;
                border: 1.5px solid #0284c7;
                border-radius: 8px;
                padding: 6px 10px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border-color: #38bdf8;
            }
        """)
        self.edit_srt.setPlaceholderText(
            "1\n00:00:00,300 --> 00:00:01,050\nGood day.\n\n"
            "2\n00:00:01,700 --> 00:00:04,000\nThis is a text-to-speech converter.\n\n"
            "3\n00:00:04,400 --> 00:00:06,850\nSubRip Subtitle (SRT) format."
        )
        self.edit_srt.textChanged.connect(self._on_srt_text_changed)
        self.edit_srt.setMinimumHeight(180)
        self.edit_srt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v_srt.addWidget(self.edit_srt, 1)

        # Bottom Bar
        h_srt_bot = QHBoxLayout()
        self.btn_clear_srt = QPushButton("🗑️ Clear")
        self.btn_clear_srt.setStyleSheet("background-color: #334155; color: #f87171; font-weight: bold; font-size: 10px; padding: 2px 8px; border-radius: 4px;")
        self.btn_clear_srt.clicked.connect(self._on_clear_srt)

        self.lbl_srt_stats = QLabel("0 បន្ទាត់ | 0 / 5000 characters")
        self.lbl_srt_stats.setStyleSheet("color: #94a3b8; font-size: 11px;")

        h_srt_bot.addWidget(self.btn_clear_srt)
        h_srt_bot.addStretch()
        h_srt_bot.addWidget(self.lbl_srt_stats)
        v_srt.addLayout(h_srt_bot)

        self.stack_text_mode.addWidget(self.widget_srt)
        v_text.addWidget(self.stack_text_mode, 1)

        self.btn_tab_plain.clicked.connect(lambda: self.stack_text_mode.setCurrentIndex(0))
        self.btn_tab_srt.clicked.connect(lambda: self.stack_text_mode.setCurrentIndex(1))

        # Batch Mode Subtitle Toolbar (Includes Quick Gender Batch Controls)
        self.batch_toolbar = QWidget()
        h_tb = QHBoxLayout(self.batch_toolbar)
        h_tb.setContentsMargins(0, 2, 0, 2)
        h_tb.setSpacing(6)

        self.btn_all_m = QPushButton("👦 ដាក់ប្រុសទាំងអស់")
        self.btn_all_m.setStyleSheet("background-color: #1e3a5f; color: #93c5fa; border: 1px solid #2563eb; font-size: 10px; padding: 4px 8px; border-radius: 4px; font-weight: bold;")
        self.btn_all_m.setToolTip("កំណត់តួអង្គប្រុស (Male) សម្រាប់គ្រប់បន្ទាត់ទាំងអស់ក្នុងតារាង")
        self.btn_all_m.clicked.connect(lambda: self.set_all_rows_gender("male"))

        self.btn_all_f = QPushButton("👧 ដាក់ស្រីទាំងអស់")
        self.btn_all_f.setStyleSheet("background-color: #4a152e; color: #f472b6; border: 1px solid #db2777; font-size: 10px; padding: 4px 8px; border-radius: 4px; font-weight: bold;")
        self.btn_all_f.setToolTip("កំណត់តួអង្គស្រី (Female) សម្រាប់គ្រប់បន្ទាត់ទាំងអស់ក្នុងតារាង")
        self.btn_all_f.clicked.connect(lambda: self.set_all_rows_gender("female"))

        self.btn_auto_g = QPushButton("⚡ Auto ប្តូរប្រុស/ស្រី")
        self.btn_auto_g.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; font-size: 10px; padding: 4px 8px; border-radius: 4px;")
        self.btn_auto_g.setToolTip("កំណត់ឆ្លាស់គ្នា ឬតាម Speaker (S1=ប្រុស, S2=ស្រី)")
        self.btn_auto_g.clicked.connect(self.auto_assign_speakers)

        self.lbl_gender_summary = QLabel("")
        self.lbl_gender_summary.setStyleSheet("font-size: 10.5px; font-weight: bold; color: #10b981;")

        h_tb.addWidget(self.btn_all_m)
        h_tb.addWidget(self.btn_all_f)
        h_tb.addWidget(self.btn_auto_g)
        h_tb.addStretch()
        h_tb.addWidget(self.lbl_gender_summary)
        v_text.addWidget(self.batch_toolbar)
        self.batch_toolbar.setVisible(False)

        # Batch Subtitle Table Widget (Displays project subtitles with clickable Male/Female toggle)
        self.table_subtitles = QTableWidget()
        self.table_subtitles.setColumnCount(7)
        self.table_subtitles.setHorizontalHeaderLabels([
            "ID", "Time", "Original Text", "Translated Text", "តួអង្គ (Voice / Gender)", "Status", "Action"
        ])
        self.table_subtitles.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_subtitles.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table_subtitles.setColumnWidth(2, 200)
        self.table_subtitles.verticalHeader().setVisible(False)
        self.table_subtitles.verticalHeader().setDefaultSectionSize(34)
        self.table_subtitles.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_subtitles.setAlternatingRowColors(True)
        self.table_subtitles.setMinimumHeight(280)
        self.table_subtitles.setStyleSheet("""
            QTableWidget {
                background-color: #0b0f19;
                alternate-background-color: #111827;
                gridline-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #1e293b;
                border-radius: 6px;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #1e3a5f;
            }
            QHeaderView::section {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                font-size: 10.5px;
                padding: 4px 6px;
                border: 1px solid #0369a1;
            }
        """)
        self.table_subtitles.itemChanged.connect(self._on_table_item_changed)
        v_text.addWidget(self.table_subtitles)
        self.table_subtitles.setVisible(False)

        self.radio_single.toggled.connect(self.on_text_mode_changed)
        self.radio_batch.toggled.connect(self.on_text_mode_changed)

        layout.addWidget(grp_text, 1)

        # --- SECTION 3: Settings ---
        self.grp_settings = QGroupBox("៣. ការកំណត់ VoxCPM2 (Model Configurations)")
        v_set = QVBoxLayout(self.grp_settings)

        h_model = QHBoxLayout()
        lbl_m = QLabel("Model Path / ID:")
        self.txt_model = QLineEdit("openbmb/VoxCPM2")
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self.on_browse_model_dir)
        h_model.addWidget(lbl_m)
        h_model.addWidget(self.txt_model, 1)
        h_model.addWidget(btn_browse_model)
        v_set.addLayout(h_model)

        h_params = QHBoxLayout()
        lbl_device = QLabel("Device:")
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto (Safe)", "CPU (Safe for 4GB GPU)", "CUDA (NVIDIA GPU)"])

        lbl_steps = QLabel("Diffusion Steps:")
        self.combo_steps = QComboBox()
        self.combo_steps.addItems(["5 (Fastest)", "10 (Balanced - Default)", "15 (High Quality)", "20 (Ultra)"])
        self.combo_steps.setCurrentIndex(1)

        lbl_cfg = QLabel("CFG Scale:")
        self.combo_cfg = QComboBox()
        self.combo_cfg.addItems(["1.5 (Creative)", "2.0 (Standard - Default)", "2.5 (Strong Adherence)"])
        self.combo_cfg.setCurrentIndex(1)

        h_params.addWidget(lbl_device)
        h_params.addWidget(self.combo_device)
        h_params.addSpacing(12)
        h_params.addWidget(lbl_steps)
        h_params.addWidget(self.combo_steps)
        h_params.addSpacing(12)
        h_params.addWidget(lbl_cfg)
        h_params.addWidget(self.combo_cfg)
        h_params.addStretch()
        v_set.addLayout(h_params)

        h_style = QHBoxLayout()
        lbl_style = QLabel("Prompt អារម្មណ៍បន្ថែម (Custom Style Prompt):")
        self.txt_style_prompt = QLineEdit()
        self.txt_style_prompt.setPlaceholderText("ឧ. និយាយបែបសុភាព ស្រទន់ មិនកាច មិនញាប់, និយាយមួយៗ...")
        h_style.addWidget(lbl_style)
        h_style.addWidget(self.txt_style_prompt, 1)
        v_set.addLayout(h_style)

        layout.addWidget(self.grp_settings)

        # --- SECTION 3: Voice Engine & Google Colab Settings ---
        self.grp_colab_settings = QGroupBox("៣. ជម្រើសសំឡេង & Google Colab Server (Voice Engine & Multi-Account Pool)")
        v_colab = QVBoxLayout(self.grp_colab_settings)
        v_colab.setSpacing(6)

        # Segmented Choice Bar for Voice Engine (Choice 1: Colab Voice Clone vs Choice 2: Piseth / Sreymom AI)
        h_engine_switcher = QHBoxLayout()
        h_engine_switcher.setSpacing(8)

        self.btn_engine_colab = QPushButton("🎙️ ជម្រើសទី ១: Clone សំឡេងមនុស្សពិត (Colab GPU)")
        self.btn_engine_colab.setCheckable(True)
        self.btn_engine_colab.setChecked(True)
        self.btn_engine_colab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_engine_colab.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_engine_colab.clicked.connect(lambda: self._set_engine_mode("colab"))

        self.btn_engine_edge = QPushButton("🗣️ ជម្រើសទី ២: សំឡេងខ្មែរ AI ធម្មតា (ពីសិទ្ធិ / ស្រីមុំ - Free & មិនបាច់ Colab)")
        self.btn_engine_edge.setCheckable(True)
        self.btn_engine_edge.setChecked(False)
        self.btn_engine_edge.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_engine_edge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_engine_edge.clicked.connect(lambda: self._set_engine_mode("edge"))

        h_engine_switcher.addWidget(self.btn_engine_colab)
        h_engine_switcher.addWidget(self.btn_engine_edge)
        v_colab.addLayout(h_engine_switcher)

        self.stack_engine_mode = QStackedWidget()

        # --- PAGE 0: Colab Multi-Account Server Settings ---
        page_colab = QWidget()
        v_page_colab = QVBoxLayout(page_colab)
        v_page_colab.setContentsMargins(0, 4, 0, 0)
        v_page_colab.setSpacing(6)

        # Row 1: Multi-Account Management, Auto-Run & Auto-Failover
        h_acc_bar = QHBoxLayout()
        lbl_acc = QLabel("👤 គណនី:")
        lbl_acc.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.combo_colab_account = QComboBox()
        self.combo_colab_account.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #0284c7;
                border-radius: 5px;
                padding: 4px 8px;
                color: #f8fafc;
                font-weight: bold;
                font-size: 11px;
                min-width: 160px;
            }
        """)
        self.combo_colab_account.currentIndexChanged.connect(self._on_colab_account_combo_changed)

        self.btn_manage_accounts = QPushButton("👥 គ្រប់គ្រងគណនី...")
        self.btn_manage_accounts.setToolTip("ចុចដើម្បីបន្ថែមគណនី Google Colab ច្រើន (Account 1, 2, 3...) ក្នុង Multi-Account Pool")
        self.btn_manage_accounts.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #38bdf8;
                border-radius: 5px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0284c7;
                color: #ffffff;
            }
        """)
        self.btn_manage_accounts.clicked.connect(self.on_open_colab_account_dialog)

        self.btn_auto_run_colab = QPushButton("🚀 Auto-Run Colab")
        self.btn_auto_run_colab.setToolTip("ចុចឱ្យ AI បើក Browser ចូលគណនី Colab ចុច Run All និងចាប់យក Link អូតូ")
        self.btn_auto_run_colab.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                border: 1px solid #10b981;
                border-radius: 5px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
        """)
        self.btn_auto_run_colab.clicked.connect(self.on_auto_run_colab)

        self.chk_auto_failover = QCheckBox("🔄 Auto-Failover (ប្តូរ Server អូតូពេលអស់ Limit)")
        self.chk_auto_failover.setChecked(True)
        self.chk_auto_failover.setStyleSheet("color: #34d399; font-weight: bold; font-size: 11px;")
        self.chk_auto_failover.toggled.connect(self._on_auto_failover_toggled)

        self.lbl_colab_status = QLabel("⚪ រង់ចាំភ្ជាប់")
        self.lbl_colab_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold; padding: 0 4px;")

        h_acc_bar.addWidget(lbl_acc)
        h_acc_bar.addWidget(self.combo_colab_account)
        h_acc_bar.addWidget(self.btn_manage_accounts)
        h_acc_bar.addWidget(self.btn_auto_run_colab)
        h_acc_bar.addWidget(self.chk_auto_failover)
        h_acc_bar.addWidget(self.lbl_colab_status)
        h_acc_bar.addStretch()
        v_page_colab.addLayout(h_acc_bar)

        # Row 2: Colab Public URL, Ping, Open Browser & Auto-Fallback
        h_colab_url = QHBoxLayout()
        lbl_curl = QLabel("🔗 Colab URL:")
        lbl_curl.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px;")
        self.txt_colab_url = QLineEdit()
        self.txt_colab_url.setPlaceholderText("Link ពី Colab (ឧ. https://xxxx.trycloudflare.com ឬ Ngrok)...")
        self.txt_colab_url.textChanged.connect(self._on_colab_url_edited)

        self.btn_ping_colab = QPushButton("🔌 Ping Server")
        self.btn_ping_colab.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; font-size: 11px; padding: 4px 10px;")
        self.btn_ping_colab.clicked.connect(self.on_ping_colab)

        self.btn_open_colab = QPushButton("🌐 Browser")
        self.btn_open_colab.setStyleSheet("background-color: #334155; color: white; font-size: 11px; padding: 4px 8px;")
        self.btn_open_colab.clicked.connect(self.on_open_colab_in_browser)

        self.chk_limit_fallback_ai = QCheckBox("✨ បើអស់ Limit ប្តូរអូតូទៅ AI:")
        self.chk_limit_fallback_ai.setChecked(True)
        self.chk_limit_fallback_ai.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 11px;")

        self.combo_fallback_voice = QComboBox()
        self.combo_fallback_voice.addItems(["👨 ពីសិទ្ធិ", "👩 ស្រីមុំ"])
        self.combo_fallback_voice.setStyleSheet("background-color: #1e293b; color: #fbbf24; font-weight: bold; border: 1px solid #f59e0b; border-radius: 4px; padding: 2px 6px; font-size: 11px;")

        h_colab_url.addWidget(lbl_curl)
        h_colab_url.addWidget(self.txt_colab_url, 1)
        h_colab_url.addWidget(self.btn_ping_colab)
        h_colab_url.addWidget(self.btn_open_colab)
        h_colab_url.addSpacing(8)
        h_colab_url.addWidget(self.chk_limit_fallback_ai)
        h_colab_url.addWidget(self.combo_fallback_voice)
        v_page_colab.addLayout(h_colab_url)

        self.stack_engine_mode.addWidget(page_colab)

        # --- PAGE 1: Khmer Edge-TTS AI Voice (Piseth / Sreymom) ---
        page_edge = QWidget()
        v_page_edge = QVBoxLayout(page_edge)
        v_page_edge.setContentsMargins(0, 4, 0, 0)
        v_page_edge.setSpacing(6)

        h_edge_v = QHBoxLayout()
        lbl_v_choice = QLabel("🎭 ជ្រើសរើសតួអង្គសំឡេង AI:")
        lbl_v_choice.setStyleSheet("color: #a78bfa; font-weight: bold; font-size: 11px;")

        self.btn_voice_piseth = QPushButton("👨 ពីសិទ្ធិ (Piseth - សំឡេងប្រុស ធម្មជាតិ)")
        self.btn_voice_piseth.setCheckable(True)
        self.btn_voice_piseth.setChecked(True)
        self.btn_voice_piseth.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_voice_piseth.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_voice_piseth.clicked.connect(lambda: self._set_edge_voice("km-KH-PisethNeural"))

        self.btn_voice_sreymom = QPushButton("👩 ស្រីមុំ (Sreymom - សំឡេងស្រី ស្រទន់)")
        self.btn_voice_sreymom.setCheckable(True)
        self.btn_voice_sreymom.setChecked(False)
        self.btn_voice_sreymom.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_voice_sreymom.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_voice_sreymom.clicked.connect(lambda: self._set_edge_voice("km-KH-SreymomNeural"))

        self.btn_preview_edge = QPushButton("▶ ចាក់ស្តាប់សំឡេងគំរូ")
        self.btn_preview_edge.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 5px 12px; font-size: 11px;")
        self.btn_preview_edge.clicked.connect(self.on_preview_edge_voice)

        h_edge_v.addWidget(lbl_v_choice)
        h_edge_v.addWidget(self.btn_voice_piseth)
        h_edge_v.addWidget(self.btn_voice_sreymom)
        h_edge_v.addWidget(self.btn_preview_edge)
        h_edge_v.addStretch()
        v_page_edge.addLayout(h_edge_v)

        lbl_edge_desc = QLabel("✨ សំឡេងខ្មែរ AI ធម្មជាតិ ឥតគិតថ្លៃ ១០០% ដំណើរការលឿនរហ័ស មិនបាច់ប្រើ Google Colab និងមិនចាំបាច់មានសំឡេងគំរូដើមឡើយ!")
        lbl_edge_desc.setStyleSheet("color: #34d399; font-size: 11px; padding: 3px 0px;")
        v_page_edge.addWidget(lbl_edge_desc)

        self.stack_engine_mode.addWidget(page_edge)
        v_colab.addWidget(self.stack_engine_mode)

        layout.addWidget(self.grp_colab_settings)
        self.grp_colab_settings.setVisible(False)

        # --- SECTION 3B: Cloud API Settings ---
        self.grp_api_settings = QGroupBox("៣. ការកំណត់ Cloud API (ElevenLabs / Website API)")
        v_api = QVBoxLayout(self.grp_api_settings)

        h_api_prov = QHBoxLayout()
        lbl_prov = QLabel("API Service:")
        self.combo_api_provider = QComboBox()
        self.combo_api_provider.addItems([
            "ElevenLabs API (elevenlabs.io)",
            "KiriTTS API (kiritts.com - Khmer AI)",
            "Custom Cloud / Website API (Khmer/REST Endpoint)"
        ])
        self.combo_api_provider.currentIndexChanged.connect(self._on_api_provider_changed)
        h_api_prov.addWidget(lbl_prov)
        h_api_prov.addWidget(self.combo_api_provider, 1)
        v_api.addLayout(h_api_prov)

        h_api_key = QHBoxLayout()
        lbl_k = QLabel("API Key:")
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_api_key.setPlaceholderText("Paste API Key ទីនេះ (xi-api-key ឬ Bearer Token)...")
        self.txt_api_key.textChanged.connect(self._on_api_key_edited)
        h_api_key.addWidget(lbl_k)
        h_api_key.addWidget(self.txt_api_key, 1)
        v_api.addLayout(h_api_key)

        h_api_extra = QHBoxLayout()
        self.lbl_voice_id = QLabel("🎭 តួអង្គ / សំឡេង (Voice):")
        self.combo_voice_id = QComboBox()
        self.combo_voice_id.setEditable(False)
        self.combo_voice_id.setMaxVisibleItems(15)
        self.combo_voice_id.setStyleSheet("""
            QComboBox {
                color: #10b981;
                font-weight: bold;
                background-color: #1e293b;
                border: 1px solid #059669;
                border-radius: 6px;
                padding: 6px 12px;
                min-height: 28px;
                font-size: 12px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid #334155;
                background-color: #0f172a;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #10b981;
                margin: auto;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f1f5f9;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
                border: 1px solid #38bdf8;
                padding: 4px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 32px;
                padding: 4px 8px;
            }
        """)
        self.combo_voice_id.currentIndexChanged.connect(self._on_voice_combo_changed)

        self.btn_fetch_voices = QPushButton("🔄")
        self.btn_fetch_voices.setToolTip("ទាញយកបញ្ជីសំឡេង Cloned ពីគណនី Website របស់អ្នកដោយស្វ័យប្រវត្តិ")
        self.btn_fetch_voices.clicked.connect(self.on_fetch_cloud_voices)

        self.lbl_endpoint = QLabel("Endpoint URL:")
        self.txt_endpoint = QLineEdit("https://api.elevenlabs.io/v1/text-to-speech")
        self.lbl_endpoint.setVisible(False)
        self.txt_endpoint.setVisible(False)

        h_api_extra.addWidget(self.lbl_voice_id)
        h_api_extra.addWidget(self.combo_voice_id, 1)
        h_api_extra.addWidget(self.btn_fetch_voices)
        h_api_extra.addWidget(self.lbl_endpoint)
        h_api_extra.addWidget(self.txt_endpoint, 1)
        v_api.addLayout(h_api_extra)

        layout.addWidget(self.grp_api_settings)
        self.grp_api_settings.setVisible(False)

        # --- SECTION 4: Execution & Progress ---
        grp_action = QGroupBox("៤. ដំណើរការ Clone & លទ្ធផល (Execution & Studio Output)")
        v_act = QVBoxLayout(grp_action)

        self.btn_start_clone = QPushButton("🎙️ ចាប់ផ្តើម Clone សំឡេង (Start Voice Clone)")
        self.btn_start_clone.setProperty("class", "btn-primary")
        self.btn_start_clone.clicked.connect(self.on_start_clone)
        v_act.addWidget(self.btn_start_clone)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(16)
        v_act.addWidget(self.progress_bar)

        self.lbl_status = QLabel("រួចរាល់សម្រាប់ Clone សំឡេង (Ready)")
        self.lbl_status.setStyleSheet("color: #94a3b8;")
        v_act.addWidget(self.lbl_status)

        # Result actions
        h_res = QHBoxLayout()
        self.btn_play_result = QPushButton("▶ ចាក់ស្តាប់សំឡេង Clone")
        self.btn_play_result.setEnabled(False)
        self.btn_play_result.clicked.connect(self.on_toggle_play_result)

        self.btn_save_audio = QPushButton("💾 រក្សាទុកសំឡេង (Save Audio)")
        self.btn_save_audio.setEnabled(False)
        self.btn_save_audio.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                border: 1px solid #0369a1;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0ea5e9;
                border-color: #38bdf8;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                border-color: #334155;
                color: #64748b;
            }
        """)
        self.btn_save_audio.clicked.connect(self.on_save_cloned_audio)

        self.btn_apply_table = QPushButton("📥 ដាក់ចូលតារាង Subtitles")
        self.btn_apply_table.setEnabled(False)
        self.btn_apply_table.setProperty("class", "btn-success")
        self.btn_apply_table.clicked.connect(self.on_apply_to_table)

        self.btn_fast_mux = QPushButton("⚡ ផ្គុំចូលវីដេអូ (-c:v copy)")
        self.btn_fast_mux.setEnabled(False)
        self.btn_fast_mux.clicked.connect(self.on_fast_mux)

        h_res.addWidget(self.btn_play_result)
        h_res.addWidget(self.btn_save_audio)
        h_res.addWidget(self.btn_apply_table)
        h_res.addWidget(self.btn_fast_mux)
        v_act.addLayout(h_res)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 1)

        # Pin Section 4 (Execution & Studio Output) permanently at the bottom so it never gets cut off
        main_layout.addWidget(grp_action)

        # Close button
        h_bottom = QHBoxLayout()
        h_bottom.addStretch()
        btn_close = QPushButton("បិទ (Close)")
        btn_close.clicked.connect(self.accept)
        h_bottom.addWidget(btn_close)
        main_layout.addLayout(h_bottom)

        # Initialize saved clone mode, Colab URL, and API provider defaults
        app_cfg = self._load_app_config()
        self.txt_colab_url.setText(app_cfg.get("colab_ngrok_url", ""))

        saved_mode_idx = app_cfg.get("selected_clone_mode_idx", 1)  # Default to Colab GPU (1)
        if 0 <= saved_mode_idx < self.combo_clone_mode.count():
            self.combo_clone_mode.setCurrentIndex(saved_mode_idx)
        self._on_mode_changed(self.combo_clone_mode.currentIndex())

        saved_prov_idx = app_cfg.get("selected_api_provider_idx", 1)  # Default to KiriTTS (1)
        if 0 <= saved_prov_idx < self.combo_api_provider.count():
            self.combo_api_provider.setCurrentIndex(saved_prov_idx)
        self._on_api_provider_changed(self.combo_api_provider.currentIndex())

                # Clean UI for Standalone Google Colab Mode
        if self.is_standalone:
            if hasattr(self, "row_female"):
                self.row_female.setVisible(False)
            if hasattr(self, "lbl_p_text"):
                self.lbl_p_text.setVisible(False)
                self.txt_prompt_text.setVisible(False)
            self.btn_ext_male.setVisible(False)
            self.btn_ext_female.setVisible(False)
            self.radio_single.setVisible(False)
            self.radio_batch.setVisible(False)
            if hasattr(self, "lbl_text_counter"):
                self.lbl_text_counter.setVisible(False)
            if hasattr(self, "batch_toolbar"):
                self.batch_toolbar.setVisible(False)
            if hasattr(self, "table_subtitles"):
                self.table_subtitles.setVisible(False)
            if hasattr(self, "btn_batch_all_male"):
                self.btn_batch_all_male.setVisible(False)
                self.btn_batch_all_female.setVisible(False)
                self.btn_batch_auto_detect.setVisible(False)
            self.btn_apply_table.setVisible(False)
            self.btn_fast_mux.setVisible(False)
            self.grp_settings.setVisible(False)
            self.grp_api_settings.setVisible(False)
            self.grp_colab_settings.setVisible(True)
            if hasattr(self, "grp_mode"):
                self.grp_mode.setVisible(False)
            self._populate_colab_accounts()
            self._set_engine_mode("colab")
            if hasattr(self, "lbl_text_counter"):
                self.lbl_text_counter.setVisible(False)
            self.edit_text.setMinimumHeight(180)
            self.edit_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            if hasattr(self, "edit_srt"):
                self.edit_srt.setMinimumHeight(180)
                self.edit_srt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.edit_text.setAcceptRichText(False)
            from PyQt6.QtGui import QFont
            font_kh = QFont("Kh Writhand", 15)
            font_kh.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            self.edit_text.setFont(font_kh)
            self.edit_text.setStyleSheet("""
                QTextEdit {
                    font-family: 'Kh Writhand', 'Kh Writehand Jrung', 'Khmer OS Battambang', 'Kantumruy Pro', sans-serif;
                    font-size: 17px;
                    color: #f8fafc;
                    background-color: #1e293b;
                    border: 1.5px solid #3b82f6;
                    border-radius: 8px;
                    padding: 8px 10px;
                    line-height: 1.5;
                }
                QTextEdit:focus {
                    border-color: #a78bfa;
                }
            """)
            if hasattr(self, "lbl_colab_info"):
                self.lbl_colab_info.setText("💡 របៀបប្រើ: ១. ចុច '🌐 បើក Google Colab' (Ctrl+F9 លើ Colab)  |  ២. Copy Link ពី Cloudflare/Ngrok មក Paste ទីនេះ រួចចាប់ផ្តើម Clone!")
                self.lbl_colab_info.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px;")
            self.edit_text.setPlaceholderText("បញ្ចូលអត្ថបទ ឬប្រយោគដែលចង់ឱ្យសំឡេង Cloned អាននិយាយនៅទីនេះ...")

    def _on_emotion_changed(self):
        if not hasattr(self, "combo_news_reader") or not hasattr(self, "combo_emotion"):
            return
        curr_emo = str(self.combo_emotion.currentData() or "")
        is_news = (curr_emo == "news_anchor")
        self.combo_news_reader.setEnabled(is_news)
        if is_news:
            self.combo_news_reader.setToolTip("ជ្រើសរើសសំឡេងគំរូអ្នកអានព័ត៌មាន System Default (មាន ៣ សំឡេងក្នុងប្រព័ន្ធ)")
        else:
            self.combo_news_reader.setToolTip("ដំណើរការតែពេលជ្រើសរើស 'អានព័ត៌មាន (News Reading)' ប៉ុណ្ណោះ")

    def _on_news_reader_selected(self, index: int):
        if not hasattr(self, "combo_news_reader") or not self.combo_news_reader.isEnabled():
            return
        filename = self.combo_news_reader.currentData()
        if not filename:
            return
        sound_path = get_system_sound_path(filename)
        if sound_path and os.path.exists(sound_path):
            if hasattr(self, "txt_ref_male"):
                self.txt_ref_male.setText(sound_path)
            if hasattr(self, "txt_ref_path"):
                self.txt_ref_path.setText(sound_path)
            if "female" in filename.lower() or "tvk" in filename.lower() or "ស្រី" in filename:
                if hasattr(self, "txt_ref_female"):
                    self.txt_ref_female.setText(sound_path)

    def _parse_srt_content(self, srt_text: str):
        import re
        blocks = re.split(r'\n\s*\n', srt_text.strip().replace('\r\n', '\n'))
        items = []
        time_pat = re.compile(r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})')
        def to_ms(h, m, s, ms):
            ms_pad = str(ms).ljust(3, '0')[:3]
            return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms_pad)

        for b in blocks:
            lines = [l.strip() for l in b.strip().split('\n') if l.strip()]
            for i, l in enumerate(lines):
                m = time_pat.search(l)
                if m:
                    s_ms = to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
                    e_ms = to_ms(m.group(5), m.group(6), m.group(7), m.group(8))
                    t = ' '.join(lines[i+1:]).strip()
                    t = re.sub(r'<[^>]+>', '', t).strip()
                    if t:
                        items.append({
                            "id": len(items) + 1,
                            "start_str": f"{m.group(1)}:{m.group(2)}:{m.group(3)},{m.group(4)}",
                            "end_str": f"{m.group(5)}:{m.group(6)}:{m.group(7)},{m.group(8)}",
                            "start_ms": s_ms,
                            "end_ms": e_ms,
                            "text": t
                        })
                    break
        return items

    def _on_srt_text_changed(self):
        text = self.edit_srt.toPlainText()
        items = self._parse_srt_content(text)
        line_count = len(items)
        char_count = len(text.strip())
        max_ms = max((p["end_ms"] for p in items), default=0)
        sec = max_ms // 1000
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        dur_str = f"{h:02d}:{m:02d}:{s:02d}"
        self.lbl_srt_dur.setText(f"⏱️ {dur_str} / 50 នាទី")
        self.lbl_srt_stats.setText(f"{line_count} បន្ទាត់ | {char_count} / 5000 characters")

    def _on_clear_srt(self):
        self.edit_srt.clear()
        self._on_srt_text_changed()

    def _on_upload_srt_file(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ជ្រើសរើសហ្វាល់ Subtitle (.srt)", "", "Subtitles (*.srt);;All Files (*)"
        )
        if file_path and os.path.exists(file_path):
            try:
                encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252"]
                content = None
                for enc in encodings:
                    try:
                        with open(file_path, "r", encoding=enc) as f:
                            content = f.read()
                        break
                    except Exception:
                        continue
                if content:
                    self.edit_srt.setPlainText(content)
                    self._on_srt_text_changed()
            except Exception as e:
                QMessageBox.warning(self, "មានបញ្ហា", f"មិនអាចអានហ្វាល់ SRT: {e}")

    def _load_app_config(self):
        try:
            import json
            cfg_p = Path("config.json")
            if cfg_p.exists():
                with open(cfg_p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_app_config_key(self, key, value):
        """Persistently saves a setting into config.json immediately."""
        try:
            import json
            cfg_p = Path("config.json")
            data = {}
            if cfg_p.exists():
                with open(cfg_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data[key] = value
            with open(cfg_p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def _save_custom_voice_to_history(self, prov_idx, voice_name):
        """Saves custom voices so they permanently appear in the dropdown."""
        voice_clean = voice_name.strip()
        if not voice_clean:
            return
        if any(base in voice_clean for base in ["Maly", "Bora", "Chanda", "Arun", "Default", "Auto Clone"]):
            return
        app_cfg = self._load_app_config()
        key = "kiritts_saved_voices" if prov_idx == 1 else "elevenlabs_saved_voices"
        saved = list(app_cfg.get(key, []))
        raw_val = voice_clean.replace("🎙️", "").strip()
        if raw_val and raw_val not in saved:
            saved.append(raw_val)
            self._save_app_config_key(key, saved)

    def _on_api_key_edited(self, text):
        prov_idx = self.combo_api_provider.currentIndex()
        if prov_idx == 0:
            self._save_app_config_key("elevenlabs_api_key", text.strip())
        elif prov_idx == 1:
            self._save_app_config_key("kiritts_api_key", text.strip())
        else:
            self._save_app_config_key("custom_api_key", text.strip())

    def _on_voice_combo_changed(self, idx):
        if idx < 0:
            return
        data = self.combo_voice_id.itemData(idx)
        prov_idx = self.combo_api_provider.currentIndex()

        if data == "__custom__":
            from PyQt6.QtWidgets import QInputDialog
            text, ok = QInputDialog.getText(
                self, "បញ្ចូល Voice ID ថ្មី",
                "បញ្ចូលឈ្មោះ ឬ Voice ID នៃសំឡេងពី Website:"
            )
            if ok and text.strip():
                clean_name = text.strip()
                new_idx = max(0, self.combo_voice_id.count() - 1)
                self.combo_voice_id.blockSignals(True)
                self.combo_voice_id.insertItem(new_idx, f"🎙️ {clean_name}", clean_name)
                self.combo_voice_id.setCurrentIndex(new_idx)
                self.combo_voice_id.blockSignals(False)
                self._save_custom_voice_to_history(prov_idx, clean_name)
                return
            else:
                self.combo_voice_id.setCurrentIndex(0)
                return

        # Auto adjust Emotion / Speed presets for dramatic character roles
        def _set_emo(val):
            idx = self.combo_emotion.findData(val)
            if idx >= 0:
                self.combo_emotion.setCurrentIndex(idx)

        def _set_spd(val):
            idx = self.combo_speed.findData(val)
            if idx >= 0:
                self.combo_speed.setCurrentIndex(idx)

        if data == "Arun_villain":
            _set_emo("angry")
        elif data == "Arun_elder":
            _set_spd(0.85)
            _set_emo("normal")
        elif data in ["Chanda", "Chanda_young"]:
            _set_emo("sad_gentle")
        elif data in ["Maly", "Bora"]:
            _set_emo("normal")
            _set_spd(1.00)

        # Save selected voice
        saved_str = str(data or self.combo_voice_id.currentText())
        if prov_idx == 0:
            self._save_app_config_key("elevenlabs_voice_id", saved_str)
        elif prov_idx == 1:
            self._save_app_config_key("kiritts_voice_id", saved_str)

    def _on_colab_url_edited(self, text):
        self._save_app_config_key("colab_ngrok_url", text.strip())

    def on_ping_colab(self):
        url = self.txt_colab_url.text().strip()
        if not url:
            QMessageBox.warning(self, "ខ្វះ URL", "សូមបញ្ចូល Colab Ngrok Public URL ជាមុនសិន។")
            return

        self.lbl_status.setText("កំពុងតេស្តតភ្ជាប់ទៅកាន់ Google Colab Server...")
        QApplication.processEvents()
        from api_voice_cloner import ping_colab_server
        try:
            info = ping_colab_server(url, timeout=8)
            self.lbl_status.setText("Colab Server Online!")
            device_str = str(info.get('device', 'cuda')).upper()
            gpu_str = str(info.get('gpu', 'NVIDIA GPU'))
            QMessageBox.information(
                self, "តភ្ជាប់ជោគជ័យ",
                f"🎉 តភ្ជាប់ទៅកាន់ Google Colab Server បានជោគជ័យ 100%!\n\n"
                f"• ឧបករណ៍ (Device): {device_str}\n"
                f"• GPU Model: {gpu_str}\n"
                f"• ស្ថានភាព: Online & Ready សម្រាប់ Clone សំឡេង VoxCPM2"
            )
        except Exception as e:
            self.lbl_status.setText("បរាជ័យក្នុងការតភ្ជាប់ Colab")
            QMessageBox.critical(self, "បរាជ័យក្នុងការតភ្ជាប់", str(e))


    def _set_engine_mode(self, mode_name: str):
        self.current_engine_mode = mode_name
        is_colab = (mode_name == "colab")
        self.btn_engine_colab.setChecked(is_colab)
        self.btn_engine_edge.setChecked(not is_colab)
        self._update_engine_button_styles()
        self.stack_engine_mode.setCurrentIndex(0 if is_colab else 1)
        if hasattr(self, "scroll") and self.scroll.verticalScrollBar():
            self.scroll.verticalScrollBar().setValue(0)

        # Update Section 1
        if hasattr(self, "txt_ref_male") and hasattr(self, "btn_browse_male"):
            self.txt_ref_male.setEnabled(is_colab)
            self.btn_browse_male.setEnabled(is_colab)
            if hasattr(self, "lbl_m"):
                if is_colab:
                    self.lbl_m.setText("🎙️ សំឡេងគំរូដើម:" if getattr(self, "is_standalone", False) else "👦 សំឡេងប្រុស:")
                    self.lbl_m.setStyleSheet("color: #cbd5e1; font-size: 11px;")
                else:
                    self.lbl_m.setText("🎙️ សំឡេងគំរូដើម (មិនបាច់ប្រើ):")
                    self.lbl_m.setStyleSheet("color: #64748b; font-size: 11px;")

        # Update Start Clone button text
        if hasattr(self, "btn_start_clone"):
            if is_colab:
                self.btn_start_clone.setText("🎙️ ចាប់ផ្តើម Clone សំឡេង (VoxCPM2 Colab)")
            else:
                v_name = "ពីសិទ្ធិ (Piseth)" if getattr(self, "selected_edge_voice", "km-KH-PisethNeural") == "km-KH-PisethNeural" else "ស្រីមុំ (Sreymom)"
                self.btn_start_clone.setText(f"🗣️ ចាប់ផ្តើមបង្កើតសំឡេង AI [{v_name}]")

    def _update_engine_button_styles(self):
        style_active = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #9333ea);
                border: 1px solid #c084fc;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 14px;
            }
        """
        style_inactive = """
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #94a3b8;
                font-weight: normal;
                font-size: 11px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f8fafc;
                border-color: #64748b;
            }
        """
        if self.btn_engine_colab.isChecked():
            self.btn_engine_colab.setStyleSheet(style_active)
            self.btn_engine_edge.setStyleSheet(style_inactive)
        else:
            self.btn_engine_colab.setStyleSheet(style_inactive)
            self.btn_engine_edge.setStyleSheet(style_active)

    def _set_edge_voice(self, voice_id: str):
        self.selected_edge_voice = voice_id
        self._update_voice_button_styles()

    def _update_voice_button_styles(self):
        cur_v = getattr(self, "selected_edge_voice", "km-KH-PisethNeural")
        is_p = (cur_v == "km-KH-PisethNeural")
        style_p_active = """
            QPushButton {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                padding: 4px 12px;
                font-size: 11px;
            }
        """
        style_p_inactive = """
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #94a3b8;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #e2e8f0;
            }
        """
        style_s_active = """
            QPushButton {
                background-color: #db2777;
                border: 1px solid #f472b6;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                padding: 4px 12px;
                font-size: 11px;
            }
        """
        style_s_inactive = """
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #94a3b8;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #e2e8f0;
            }
        """
        self.btn_voice_piseth.setStyleSheet(style_p_active if is_p else style_p_inactive)
        self.btn_voice_sreymom.setStyleSheet(style_s_active if not is_p else style_s_inactive)
        if hasattr(self, "btn_start_clone") and getattr(self, "current_engine_mode", "colab") == "edge":
            v_title = "ពីសិទ្ធិ (Piseth)" if is_p else "ស្រីមុំ (Sreymom)"
            self.btn_start_clone.setText(f"🗣️ ចាប់ផ្តើមបង្កើតសំឡេង AI [{v_title}]")

    def _populate_colab_accounts(self):
        try:
            from colab_auto_runner import load_colab_auto_config
            self.colab_cfg = load_colab_auto_config()
            self.combo_colab_account.blockSignals(True)
            self.combo_colab_account.clear()
            active_id = self.colab_cfg.get("active_account_id", 1)
            sel_idx = 0
            for idx, a in enumerate(self.colab_cfg.get("accounts", [])):
                name = a.get("name", f"Account {a.get('id')}")
                email = a.get("email", "")
                disp = f"{name} ({email})" if email else name
                self.combo_colab_account.addItem(disp, a.get("id"))
                if a.get("id") == active_id:
                    sel_idx = idx
            self.combo_colab_account.setCurrentIndex(sel_idx)
            self.combo_colab_account.blockSignals(False)

            self.chk_auto_failover.blockSignals(True)
            self.chk_auto_failover.setChecked(self.colab_cfg.get("auto_failover_enabled", True))
            self.chk_auto_failover.blockSignals(False)
        except Exception as e:
            print("Error populating colab accounts:", e)

    def _on_colab_account_combo_changed(self, idx: int):
        if idx < 0:
            return
        try:
            from colab_auto_runner import load_colab_auto_config, save_colab_auto_config
            acc_id = self.combo_colab_account.currentData()
            if acc_id:
                cfg = load_colab_auto_config()
                cfg["active_account_id"] = acc_id
                save_colab_auto_config(cfg)
                self.colab_cfg = cfg
                for a in cfg.get("accounts", []):
                    if a.get("id") == acc_id:
                        nb_u = a.get("notebook_url", "")
                        if nb_u:
                            self._save_app_config_key("colab_notebook_url", nb_u)
                        break
        except Exception as e:
            print("Error switching colab account:", e)

    def on_open_colab_account_dialog(self):
        try:
            from colab_auto_runner import ColabAccountDialog
            dlg = ColabAccountDialog(self)
            if dlg.exec():
                self._populate_colab_accounts()
                if getattr(dlg, "start_immediately", False):
                    self.on_auto_run_colab()
        except Exception as e:
            QMessageBox.warning(self, "ការកំណត់គណនី", f"មិនអាចបើកផ្ទាំងកំណត់គណនីបានទេ:\n{e}")

    def on_auto_run_colab(self):
        try:
            from colab_auto_runner import (
                load_colab_auto_config, ColabAccountDialog, ColabAutoRunWorker,
                get_active_account
            )
            cfg = load_colab_auto_config()
            curr_acc = get_active_account(cfg)
            nb_url = (curr_acc.get("notebook_url") or "").strip()
            if not nb_url or nb_url == "https://colab.research.google.com":
                dlg = ColabAccountDialog(self)
                if not dlg.exec():
                    return
                cfg = load_colab_auto_config()
                curr_acc = get_active_account(cfg)

            target_acc_id = curr_acc.get("id", 1)

            if hasattr(self, "colab_auto_worker") and self.colab_auto_worker is not None and self.colab_auto_worker.isRunning():
                QMessageBox.information(self, "Auto-Run Colab", "ប្រព័ន្ធកំពុងដំណើរការ Auto-Run Colab រួចហើយ សូមរង់ចាំបន្តិច...")
                return

            self.btn_auto_run_colab.setEnabled(False)
            self.btn_auto_run_colab.setText("⏳ កំពុងបើក Server...")
            self.lbl_colab_status.setText("⚙️ កំពុងដំណើរការ...")
            self.lbl_colab_status.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")

            self.colab_auto_worker = ColabAutoRunWorker(
                account_id=target_acc_id,
                existing_url=self.txt_colab_url.text().strip(),
                allow_failover=self.chk_auto_failover.isChecked(),
                cdp_port=int(cfg.get("cdp_port", 9233)),
            )
            self.colab_auto_worker.status_badge_signal.connect(self._on_auto_colab_status_badge)
            self.colab_auto_worker.account_switched_signal.connect(self._on_auto_colab_account_switched)
            self.colab_auto_worker.server_ready_signal.connect(self._on_auto_colab_server_ready)
            self.colab_auto_worker.failed_signal.connect(self._on_auto_colab_failed)
            self.colab_auto_worker.start()
        except Exception as e:
            self.btn_auto_run_colab.setEnabled(True)
            self.btn_auto_run_colab.setText("🚀 Auto-Run Colab")
            QMessageBox.critical(self, "Auto-Run Colab Error", f"មានបញ្ហាក្នុងការបើក Colab Server:\n{e}")

    def _on_auto_colab_status_badge(self, text: str, color_hex: str):
        self.lbl_colab_status.setText(text)
        self.lbl_colab_status.setStyleSheet(f"color: {color_hex}; font-size: 11px; font-weight: bold; padding: 0 4px;")

    def _on_auto_colab_account_switched(self, account_name: str, reason: str):
        self._on_auto_colab_status_badge(f"🔄 ប្តូរទៅ {account_name}", "#f59e0b")
        self.lbl_status.setText(f"Auto-Failover: កំពុងប្តូរទៅ [{account_name}]...")

    def _on_auto_colab_server_ready(self, public_url: str, gpu_info: str):
        self.btn_auto_run_colab.setEnabled(True)
        self.btn_auto_run_colab.setText("✅ Colab ដើររួចរាល់")
        self.txt_colab_url.setText(public_url)
        self._save_app_config_key("colab_ngrok_url", public_url)
        self._on_auto_colab_status_badge(f"✅ ភ្ជាប់ជោគជ័យ ({gpu_info[:16]})", "#22c55e")
        self.lbl_status.setText("Google Colab ភ្ជាប់ជោគជ័យ 100%!")
        self.on_ping_colab()

    def _on_auto_colab_failed(self, err_msg: str):
        self.btn_auto_run_colab.setEnabled(True)
        self.btn_auto_run_colab.setText("🚀 Auto-Run Colab")
        self._on_auto_colab_status_badge("❌ បរាជ័យ", "#ef4444")
        reply = QMessageBox.question(
            self, "Colab Server បរាជ័យ",
            f"ការតភ្ជាប់ Colab Server មិនបានសម្រេច:\n{err_msg}\n\n"
            f"តើបងចង់ប្តូរទៅប្រើ [🗣️ សំឡេង AI ពីសិទ្ធិ / ស្រីមុំ] ជំនួសវិញភ្លាមៗដែរឬទេ? (ដំណើរការបានភ្លាម មិនបាច់ Colab)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_engine_mode("edge")

    def on_preview_edge_voice(self):
        voice_id = getattr(self, "selected_edge_voice", "km-KH-PisethNeural")
        is_male = ("Piseth" in voice_id)
        phrase = "សួស្តីបង ខ្ញុំបាទឈ្មោះ ពីសិទ្ធិ។" if is_male else "សួស្តីបង ខ្ញុំចាសឈ្មោះ ស្រីមុំ។"
        preview_wav = str(CACHE_DIR / f"preview_{'piseth' if is_male else 'sreymom'}.wav")
        if not os.path.exists(preview_wav):
            try:
                import asyncio, edge_tts, subprocess, time
                clean = phrase
                temp_mp3 = preview_wav + ".mp3"
                last_err = None
                for attempt in range(4):
                    try:
                        async def _r():
                            comm = edge_tts.Communicate(clean, voice_id)
                            await comm.save(temp_mp3)
                        asyncio.run(_r())
                        if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 300:
                            last_err = None
                            break
                    except Exception as ex:
                        last_err = ex
                        time.sleep(1.0 * (attempt + 1))
                if last_err and (not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) <= 300):
                    raise last_err
                cmd = [FFMPEG_EXE, "-y", "-i", temp_mp3, "-ar", "48000", preview_wav]
                silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            except Exception as e:
                QMessageBox.warning(self, "Preview Error", f"មិនអាចបង្កើតសំឡេងគំរូបានទេ (អាចមកពីអ៊ីនធឺណិតរអាក់រអួល):\n{e}")
                return
        if os.path.exists(preview_wav):
            self.player.setSource(QUrl.fromLocalFile(preview_wav))
            self.player.play()

    def _on_auto_failover_toggled(self, checked: bool):
        try:
            from colab_auto_runner import load_colab_auto_config, save_colab_auto_config
            cfg = load_colab_auto_config()
            cfg["auto_failover_enabled"] = bool(checked)
            save_colab_auto_config(cfg)
        except Exception as e:
            print("Error toggling failover:", e)

    def on_open_colab_in_browser(self):
        import webbrowser
        app_cfg = self._load_app_config()
        notebook_url = app_cfg.get("colab_notebook_url", "https://colab.research.google.com")
        webbrowser.open(notebook_url)
        self.lbl_status.setText("បានបើក Google Colab លើ Browser រួចរាល់!")

    def on_set_colab_notebook_url(self):
        from PyQt6.QtWidgets import QInputDialog
        app_cfg = self._load_app_config()
        current_url = app_cfg.get("colab_notebook_url", "https://colab.research.google.com")
        text, ok = QInputDialog.getText(
            self, "កំណត់ Link Google Colab Notebook",
            "បញ្ចូល Link នៃ Colab Notebook ផ្ទាល់ខ្លួនរបស់អ្នក (Google Drive Colab Link):\n(ឧ. https://colab.research.google.com/drive/xxx...)",
            text=current_url
        )
        if ok and text.strip():
            self._save_app_config_key("colab_notebook_url", text.strip())
            QMessageBox.information(
                self, "រក្សាទុកជោគជ័យ",
                "បានរក្សាទុក Link Notebook រួចរាល់!\nពេលក្រោយចុចប៊ូតុង 'បើក Google Colab' វានឹងរត់ទៅកាន់ Notebook របស់អ្នកភ្លាម។"
            )

    def _on_mode_changed(self, idx):
        if getattr(self, "is_standalone", False):
            self.grp_settings.setVisible(False)
            self.grp_colab_settings.setVisible(True)
            self.grp_api_settings.setVisible(False)
            return
        self._save_app_config_key("selected_clone_mode_idx", idx)
        self.grp_settings.setVisible(idx == 0)
        self.grp_colab_settings.setVisible(idx == 1)
        self.grp_api_settings.setVisible(idx == 2)

    def _on_api_provider_changed(self, idx):
        self._save_app_config_key("selected_api_provider_idx", idx)
        app_cfg = self._load_app_config()
        self.combo_voice_id.blockSignals(True)
        self.combo_voice_id.clear()

        if idx == 0:  # ElevenLabs
            self.lbl_voice_id.setVisible(True)
            self.combo_voice_id.setVisible(True)
            self.btn_fetch_voices.setVisible(True)
            self.lbl_endpoint.setVisible(False)
            self.txt_endpoint.setVisible(False)
            self.txt_endpoint.setText("https://api.elevenlabs.io/v1/text-to-speech")

            # Load saved ElevenLabs API Key
            self.txt_api_key.blockSignals(True)
            self.txt_api_key.setText(app_cfg.get("elevenlabs_api_key", ""))
            self.txt_api_key.blockSignals(False)

            # Populate ElevenLabs voice options
            self.combo_voice_id.addItem("🤖 Auto Clone ពីសំឡេងគំរូ (Default)", "")
            self.combo_voice_id.addItem("👨 តួឯកប្រុស (Male Lead)", "male_lead")
            self.combo_voice_id.addItem("👩 តួឯកស្រី (Female Lead)", "female_lead")
            self.combo_voice_id.addItem("👨 តួរងប្រុស (Male Supporting)", "male_sup")
            self.combo_voice_id.addItem("👩 តួរងស្រី (Female Supporting)", "female_sup")

            saved = app_cfg.get("elevenlabs_saved_voices", [])
            for sv in saved:
                if sv and sv not in ["Auto Clone ពីសំឡេងគំរូ (Default)", "male_lead", "female_lead", "male_sup", "female_sup"]:
                    self.combo_voice_id.addItem(f"🎙️ {sv}", sv)

            self.combo_voice_id.addItem("➕ បញ្ចូល Voice ID ថ្មី (Custom Voice ID)...", "__custom__")

            last_v = app_cfg.get("elevenlabs_voice_id", "")
            if last_v:
                matched_idx = -1
                for i in range(self.combo_voice_id.count()):
                    d = self.combo_voice_id.itemData(i)
                    t = self.combo_voice_id.itemText(i)
                    if last_v and (last_v == d or last_v == t or (d and d in last_v)):
                        matched_idx = i
                        break
                if matched_idx >= 0:
                    self.combo_voice_id.setCurrentIndex(matched_idx)

        elif idx == 1:  # KiriTTS
            self.lbl_voice_id.setVisible(True)
            self.combo_voice_id.setVisible(True)
            self.btn_fetch_voices.setVisible(True)
            self.lbl_endpoint.setVisible(True)
            self.txt_endpoint.setVisible(True)
            self.txt_endpoint.setText("https://api.kiritts.com/v1/audio/speech")

            # Load saved KiriTTS API Key
            self.txt_api_key.blockSignals(True)
            self.txt_api_key.setText(app_cfg.get("kiritts_api_key", ""))
            self.txt_api_key.blockSignals(False)

            # Populate KiriTTS voice options (Standard Dramatic Roles + Custom Clones)
            self.combo_voice_id.addItem("👨 តួឯកប្រុស (Male Lead - Bora ធម្មជាតិ)", "Bora")
            self.combo_voice_id.addItem("👩 តួឯកស្រី (Female Lead - Maly ធម្មជាតិ)", "Maly")
            self.combo_voice_id.addItem("👨 តួរងប្រុស (Male Supporting - Arun ម៉ឺងម៉ាត់)", "Arun")
            self.combo_voice_id.addItem("👩 តួរងស្រី (Female Supporting - Chanda ស្រទន់)", "Chanda")
            self.combo_voice_id.addItem("🎙️ តួចិត្តអាក្រក់ / តួកាច (Villain / Evil - Arun Firm)", "Arun_villain")
            self.combo_voice_id.addItem("👵 តួមនុស្សចាស់ / យាយតា (Elderly Voice - Arun Calm)", "Arun_elder")
            self.combo_voice_id.addItem("🧒 តួក្មេង / យុវវ័យ (Young / Soft - Chanda Warm)", "Chanda_young")

            saved = app_cfg.get("kiritts_saved_voices", [])
            for sv in saved:
                if sv and sv not in ["Maly", "Bora", "Chanda", "Arun", "Arun_villain", "Arun_elder", "Chanda_young"]:
                    self.combo_voice_id.addItem(f"🎙️ {sv}", sv)

            self.combo_voice_id.addItem("➕ បញ្ចូល Voice ID / Clone ថ្មី (Custom Voice ID)...", "__custom__")

            last_v = app_cfg.get("kiritts_voice_id", "Bora")
            matched_idx = -1
            for i in range(self.combo_voice_id.count()):
                d = self.combo_voice_id.itemData(i)
                t = self.combo_voice_id.itemText(i)
                if last_v and (last_v == d or last_v == t or (d and d in last_v)):
                    matched_idx = i
                    break
            if matched_idx >= 0:
                self.combo_voice_id.setCurrentIndex(matched_idx)
            else:
                self.combo_voice_id.setCurrentIndex(0)

        else:  # Custom
            self.lbl_voice_id.setVisible(False)
            self.combo_voice_id.setVisible(False)
            self.btn_fetch_voices.setVisible(False)
            self.lbl_endpoint.setVisible(True)
            self.txt_endpoint.setVisible(True)

            self.txt_api_key.blockSignals(True)
            self.txt_api_key.setText(app_cfg.get("custom_api_key", ""))
            self.txt_api_key.blockSignals(False)

        self.combo_voice_id.blockSignals(False)

    def on_fetch_cloud_voices(self):
        """Fetch available voices directly from KiriTTS or ElevenLabs account."""
        prov_idx = self.combo_api_provider.currentIndex()
        api_k = self.txt_api_key.text().strip()
        if not api_k:
            QMessageBox.warning(self, "ខ្វះ API Key", "សូមបញ្ចូល API Key ជាមុនសិន ទើបអាចទាញយកបញ្ជីសំឡេងបាន។")
            return

        self.lbl_status.setText("កំពុងទាញយកបញ្ជីសំឡេងពី Server...")
        if prov_idx == 1:  # KiriTTS
            from api_voice_cloner import fetch_kiritts_voices
            voices = fetch_kiritts_voices(api_k)
            if voices:
                app_cfg = self._load_app_config()
                saved = app_cfg.get("kiritts_saved_voices", [])
                for v in voices:
                    if v not in saved:
                        saved.append(v)
                self._save_app_config_key("kiritts_saved_voices", saved)
                self._on_api_provider_changed(1)
                QMessageBox.information(self, "ទាញយកជោគជ័យ", f"🎉 បានទាញយកសំឡេងចំនួន {len(voices)} ពីគណនី KiriTTS របស់អ្នក!")
            else:
                QMessageBox.information(self, "លទ្ធផល", "មិនមានសំឡេង Cloned បន្ថែមក្នុងគណនី KiriTTS នេះទេ។")
        elif prov_idx == 0:  # ElevenLabs
            from api_voice_cloner import fetch_elevenlabs_voices
            voices = fetch_elevenlabs_voices(api_k)
            if voices:
                app_cfg = self._load_app_config()
                saved = app_cfg.get("elevenlabs_saved_voices", [])
                for v_name, v_id in voices:
                    if v_name not in saved:
                        saved.append(v_name)
                self._save_app_config_key("elevenlabs_saved_voices", saved)
                self._on_api_provider_changed(0)
                QMessageBox.information(self, "ទាញយកជោគជ័យ", f"🎉 បានទាញយកសំឡេងចំនួន {len(voices)} ពីគណនី ElevenLabs!")

    def on_browse_ref_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ជ្រើសរើសសំឡេងគំរូ (Reference Audio)", "",
            "Audio Files (*.wav *.mp3 *.m4a *.aac *.flac)"
        )
        if file_path:
            self.txt_ref_path.setText(file_path)

    def on_browse_model_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "ជ្រើសរើស Local Folder នៃ VoxCPM2 Model")
        if folder:
            self.txt_model.setText(folder)

    def on_browse_ref_audio(self, gender="male"):
        f, _ = QFileDialog.getOpenFileName(
            self,
            f"ជ្រើសរើសសំឡេងគំរូ ({'មនុស្សប្រុស' if gender == 'male' else 'មនុស្សស្រី'})",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.aac *.flac)"
        )
        if f:
            if gender == "female":
                self.txt_ref_female.setText(f)
            else:
                self.txt_ref_male.setText(f)
            if hasattr(self, "combo_news_reader"):
                self.combo_news_reader.blockSignals(True)
                self.combo_news_reader.setCurrentIndex(0)
                self.combo_news_reader.blockSignals(False)

    def on_extract_from_video(self, gender="male"):
        """Extracts 5s clean voice reference directly from the loaded video."""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(self, "មិនមានវីដេអូ", "សូម Import វីដេអូជាមុនសិន។")
            return

        out_ref = f"output/extracted_ref_voice_{gender}.wav"
        Path(out_ref).parent.mkdir(parents=True, exist_ok=True)
        ss_time = "00:00:08" if gender == "female" else "00:00:02"
        cmd = [
            FFMPEG_EXE, "-y",
            "-ss", ss_time,
            "-i", self.current_video_path,
            "-t", "5",
            "-ac", "1",
            "-ar", "16000",
            out_ref
        ]
        res = silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(out_ref):
            full_p = str(Path(out_ref).resolve())
            if gender == "female":
                self.txt_ref_female.setText(full_p)
            else:
                self.txt_ref_male.setText(full_p)
            self.lbl_status.setText(f"ទាញសំឡេង 5 វិនាទី ({'ប្រុស' if gender == 'male' else 'ស្រី'}) បានជោគជ័យ!")
        else:
            QMessageBox.warning(self, "ទាញយកបរាជ័យ", "មិនអាចទាញសំឡេងពីវីដេអូបានទេ។")

    def on_toggle_play_ref(self, gender="male"):
        path = self.txt_ref_female.text().strip() if gender == "female" else self.txt_ref_male.text().strip()
        btn = self.btn_play_female if gender == "female" else self.btn_play_male
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "គ្មានសំឡេង", f"សូមជ្រើសរើសឯកសារសំឡេងគំរូ ({'មនុស្សប្រុស' if gender == 'male' else 'មនុស្សស្រី'}) ជាមុនសិន។")
            return
        if self.player_ref.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player_ref.stop()
            btn.setText("▶ ស្តាប់")
        else:
            self.player_ref.setSource(QUrl.fromLocalFile(path))
            self.player_ref.play()
            btn.setText("⏸️ ផ្អាក")

    def _on_player_res_playback_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.btn_play_result.setText("▶ ចាក់ស្តាប់សំឡេង Clone")

    def on_toggle_play_result(self):
        if not self.output_cloned_wav or not os.path.exists(self.output_cloned_wav):
            QMessageBox.warning(self, "មិនទាន់មានសំឡេង", "មិនទាន់មានឯកសារសំឡេង Clone នៅឡើយទេ។ សូមចុច 'ចាប់ផ្តើម Clone សំឡេង' ជាមុនសិន។")
            return
        if self.player_res.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player_res.pause()
            self.btn_play_result.setText("▶ ចាក់ស្តាប់សំឡេង Clone")
        else:
            self.player_res.setSource(QUrl.fromLocalFile(self.output_cloned_wav))
            self.player_res.play()
            self.btn_play_result.setText("⏸️ ផ្អាក")

    def on_text_mode_changed(self):
        if self.radio_batch.isChecked():
            self.is_batch_mode = True
            self.edit_text.setVisible(False)
            self.batch_toolbar.setVisible(True)
            self.table_subtitles.setVisible(True)
            self.populate_subtitles_table()
            sub_count = len(self.subtitles) if self.subtitles else 0
            self.lbl_text_counter.setText(f"សរុប {sub_count} ឃ្លាក្នុងតារាង Subtitles")
        else:
            self.is_batch_mode = False
            self.table_subtitles.setVisible(False)
            self.batch_toolbar.setVisible(False)
            self.edit_text.setVisible(True)
            self.edit_text.setPlainText(self.active_text)
            self.lbl_text_counter.setText("អត្ថបទបន្ទាត់បច្ចុប្បន្ន")

    def populate_subtitles_table(self):
        if not hasattr(self, "table_subtitles") or not self.subtitles:
            return

        self.table_subtitles.blockSignals(True)
        self.table_subtitles.setRowCount(len(self.subtitles))
        self.table_subtitles.verticalHeader().setDefaultSectionSize(34)
        for row_idx, item in enumerate(self.subtitles):
            # 1. ID
            id_item = QTableWidgetItem(str(getattr(item, "item_id", row_idx + 1) or (row_idx + 1)))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_subtitles.setItem(row_idx, 0, id_item)

            # 2. Time
            s_t = item.start_time.split(",")[0] if (item.start_time and "," in item.start_time) else (item.start_time or "00:00:00")
            e_t = item.end_time.split(",")[0] if (item.end_time and "," in item.end_time) else (item.end_time or "00:00:00")
            time_str = f"{s_t} - {e_t}" if s_t != e_t else s_t
            time_item = QTableWidgetItem(time_str)
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_subtitles.setItem(row_idx, 1, time_item)

            # 3. Original Text
            orig_item = QTableWidgetItem(item.original_text or "")
            orig_item.setFlags(orig_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            orig_item.setForeground(QColor("#94a3b8"))
            self.table_subtitles.setItem(row_idx, 2, orig_item)

            # 4. Translated Text (Editable directly)
            trans_text = item.translated_text if item.translated_text else item.original_text
            trans_item = QTableWidgetItem(trans_text or "")
            trans_item.setForeground(QColor("#f8fafc"))
            self.table_subtitles.setItem(row_idx, 3, trans_item)

            # 5. Gender / Voice Toggle Button (Red circled feature)
            btn_gender = self._create_gender_button(row_idx, item)
            self.table_subtitles.setCellWidget(row_idx, 4, btn_gender)

            # 6. Status
            stat_text = getattr(item, "status", "Ready") or "Ready"
            stat_item = QTableWidgetItem(stat_text)
            stat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_item.setFlags(stat_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if stat_text == "Cloned":
                stat_item.setForeground(QColor("#10b981"))
            elif "TTS" in stat_text:
                stat_item.setForeground(QColor("#facc15"))
            else:
                stat_item.setForeground(QColor("#38bdf8"))
            self.table_subtitles.setItem(row_idx, 5, stat_item)

            # 7. Action Cell Widget (Play line + Clone single line)
            act_widget = self._create_row_action_widget(row_idx, item)
            self.table_subtitles.setCellWidget(row_idx, 6, act_widget)

        self.table_subtitles.blockSignals(False)
        self._update_gender_count_summary()

    def _create_gender_button(self, row_idx: int, item) -> QWidget:
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn = QPushButton()
        self._style_gender_button(btn, item)
        btn.clicked.connect(lambda _, r=row_idx, b=btn: self._on_toggle_row_gender(r, b))

        lay.addWidget(btn)
        return container

    def _style_gender_button(self, btn: QPushButton, item):
        gender = getattr(item, "gender", "male") or "male"
        spk = getattr(item, "speaker", "")
        spk_str = f" ({spk.replace('Speaker ', 'S')})" if spk else ""

        if gender == "female":
            btn.setText(f"👧 ស្រី{spk_str}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a152e;
                    color: #f472b6;
                    border: 1px solid #db2777;
                    border-radius: 3px;
                    padding: 2px 7px;
                    font-size: 9.5px;
                    font-weight: 600;
                    min-height: 22px;
                    max-height: 24px;
                }
                QPushButton:hover {
                    background-color: #db2777;
                    color: #ffffff;
                }
            """)
            btn.setToolTip("សំឡេងមនុស្សស្រី (Female) - ចុចដើម្បីប្តូរទៅប្រុស")
        else:
            btn.setText(f"👦 ប្រុស{spk_str}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e3a5f;
                    color: #93c5fa;
                    border: 1px solid #2563eb;
                    border-radius: 3px;
                    padding: 2px 7px;
                    font-size: 9.5px;
                    font-weight: 600;
                    min-height: 22px;
                    max-height: 24px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                    color: #ffffff;
                }
            """)
            btn.setToolTip("សំឡេងមនុស្សប្រុស (Male) - ចុចដើម្បីប្តូរទៅស្រី")

    def _on_toggle_row_gender(self, row_idx: int, btn: QPushButton):
        if 0 <= row_idx < len(self.subtitles):
            item = self.subtitles[row_idx]
            cur = getattr(item, "gender", "male") or "male"
            item.gender = "female" if cur == "male" else "male"
            item.gender_locked = True
            self._style_gender_button(btn, item)
            self._update_gender_count_summary()

    def set_all_rows_gender(self, target_gender: str):
        if not self.subtitles:
            return
        for r, item in enumerate(self.subtitles):
            item.gender = target_gender
            item.gender_locked = True
            if hasattr(self, "table_subtitles") and self.table_subtitles.rowCount() > r:
                btn_container = self.table_subtitles.cellWidget(r, 4)
                if btn_container:
                    btn = btn_container.findChild(QPushButton)
                    if btn:
                        self._style_gender_button(btn, item)
        self._update_gender_count_summary()

    def auto_assign_speakers(self):
        if not self.subtitles:
            return
        for r, item in enumerate(self.subtitles):
            spk = getattr(item, "speaker", "").lower()
            if "2" in spk or "female" in spk or "ស្រី" in spk:
                item.gender = "female"
            elif "1" in spk or "male" in spk or "ប្រុស" in spk:
                item.gender = "male"
            else:
                item.gender = "male" if (r % 2 == 0) else "female"

            if hasattr(self, "table_subtitles") and self.table_subtitles.rowCount() > r:
                btn_container = self.table_subtitles.cellWidget(r, 4)
                if btn_container:
                    btn = btn_container.findChild(QPushButton)
                    if btn:
                        self._style_gender_button(btn, item)
        self._update_gender_count_summary()

    def _update_gender_count_summary(self):
        if not self.subtitles:
            if hasattr(self, "lbl_gender_summary"):
                self.lbl_gender_summary.setText("")
            return
        m_count = sum(1 for s in self.subtitles if getattr(s, "gender", "male") == "male")
        f_count = sum(1 for s in self.subtitles if getattr(s, "gender", "male") == "female")
        total = len(self.subtitles)
        if hasattr(self, "lbl_gender_summary"):
            self.lbl_gender_summary.setText(f"👦 ប្រុស: {m_count}  |  👧 ស្រី: {f_count}  (សរុប {total} ឃ្លា)")

    def _create_row_action_widget(self, row_idx: int, item) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        has_audio = bool(item.audio_path and os.path.exists(item.audio_path) and os.path.getsize(item.audio_path) > 100)
        btn_play = QPushButton("▶ ស្តាប់")
        btn_play.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                border: 1px solid #047857;
                font-size: 9.5px;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 3px;
                min-height: 22px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }
        """)
        btn_play.setEnabled(has_audio)
        btn_play.clicked.connect(lambda _, r=row_idx: self._play_single_row_audio(r))

        btn_clone_row = QPushButton("🎙️ Clone")
        btn_clone_row.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                font-size: 9.5px;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 3px;
                min-height: 22px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: #0284c7;
                color: #ffffff;
                border-color: #38bdf8;
            }
        """)
        btn_clone_row.setToolTip("Clone សំឡេងតែមួយបន្ទាត់នេះភ្លាមៗ")
        btn_clone_row.clicked.connect(lambda _, r=row_idx: self._clone_single_row(r))

        lay.addWidget(btn_play)
        lay.addWidget(btn_clone_row)
        return w

    def _play_single_row_audio(self, row_idx: int):
        if 0 <= row_idx < len(self.subtitles):
            item = self.subtitles[row_idx]
            if item.audio_path and os.path.exists(item.audio_path):
                self.player_res.setSource(QUrl.fromLocalFile(item.audio_path))
                self.player_res.play()

    def _clone_single_row(self, row_idx: int):
        if 0 <= row_idx < len(self.subtitles):
            item = self.subtitles[row_idx]
            t = (item.translated_text if item.translated_text else item.original_text).strip()
            if not t:
                return

            gender = getattr(item, "gender", "male").lower()
            ref_f = self.txt_ref_female.text().strip() if hasattr(self, "txt_ref_female") else ""
            ref_m = self.txt_ref_male.text().strip() if hasattr(self, "txt_ref_male") else ""
            if gender == "female":
                ref_path = ref_f if (ref_f and os.path.exists(ref_f)) else ref_m
            else:
                ref_path = ref_m if (ref_m and os.path.exists(ref_m)) else ref_f

            if not ref_path or not os.path.exists(ref_path):
                QMessageBox.warning(self, "ខ្វះសំឡេងគំរូ", "សូមជ្រើសរើសសំឡេងគំរូ (ប្រុស ឬ ស្រី) ជាមុនសិន!")
                return

            self.lbl_status.setText(f"កំពុង Clone បន្ទាត់ទី {row_idx + 1} ({'👧 ស្រី' if gender == 'female' else '👦 ប្រុស'})...")

            import uuid
            # CACHE_DIR globally defined
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Purge previous audio files for this item_id
            for old_f in CACHE_DIR.glob(f"single_row_{item.item_id}_*.wav"):
                try:
                    old_f.unlink(missing_ok=True)
                except Exception:
                    pass
            for old_f in CACHE_DIR.glob(f"cloned_line_{item.item_id}_*.wav"):
                try:
                    old_f.unlink(missing_ok=True)
                except Exception:
                    pass
            for old_f in CACHE_DIR.glob(f"voice_row_{item.item_id}_*.mp3"):
                try:
                    old_f.unlink(missing_ok=True)
                except Exception:
                    pass

            out_file = str(CACHE_DIR / f"cloned_line_{item.item_id}.wav")

            mode_idx = self.combo_clone_mode.currentIndex()
            mode = ["voxcpm", "colab_voxcpm", "cloud_api", "fast_neural"][mode_idx]
            colab_url = self.txt_colab_url.text().strip() if hasattr(self, "txt_colab_url") else ""
            if mode == "colab_voxcpm" and not colab_url:
                QMessageBox.warning(self, "ខ្វះ Colab URL", "សូមបញ្ចូល Colab Ngrok Public URL ជាមុនសិន!")
                return

            prov_idx = self.combo_api_provider.currentIndex() if hasattr(self, "combo_api_provider") else 0
            api_prov = "elevenlabs" if prov_idx == 0 else ("kiritts" if prov_idx == 1 else "custom")
            api_k = self.txt_api_key.text().strip() if hasattr(self, "txt_api_key") else ""
            api_ep = self.txt_endpoint.text().strip() if hasattr(self, "txt_endpoint") else ""
            v_data = self.combo_voice_id.itemData(self.combo_voice_id.currentIndex()) if hasattr(self, "combo_voice_id") else None
            raw_v = self.combo_voice_id.currentText().strip() if hasattr(self, "combo_voice_id") else ""
            v_id = str(v_data if v_data is not None else raw_v)
            if v_id.startswith("🎙️"):
                v_id = v_id.replace("🎙️", "").strip()
            speed_val = float(self.combo_speed.currentData() or 1.0)
            emotion_val = str(self.combo_emotion.currentData() or "normal")
            style_prompt_val = self.txt_style_prompt.text().strip() if hasattr(self, "txt_style_prompt") else ""
            timesteps = {0: 5, 1: 10, 2: 15, 3: 20}.get(self.combo_steps.currentIndex(), 10) if hasattr(self, "combo_steps") else 10
            cfg_val = {0: 1.5, 1: 2.0, 2: 2.5}.get(self.combo_cfg.currentIndex(), 2.0) if hasattr(self, "combo_cfg") else 2.0
            model_path = self.txt_model.text().strip() if hasattr(self, "txt_model") else "openbmb/VoxCPM2"
            prompt_txt = self.txt_prompt_text.text().strip() if hasattr(self, "txt_prompt_text") else None

            self.worker = VoiceCloneWorker(
                text=t,
                ref_audio_path=ref_path,
                output_wav_path=out_file,
                mode=mode,
                prompt_text=prompt_txt,
                model_path=model_path,
                cfg_value=cfg_val,
                timesteps=timesteps,
                device="auto",
                api_provider=api_prov,
                api_key=api_k,
                api_endpoint=api_ep,
                voice_id=v_id,
                speed=speed_val,
                emotion=emotion_val,
                style_prompt=style_prompt_val,
                batch_items=[],
                colab_url=colab_url,
                ref_audio_path_female=ref_f if (ref_f and os.path.exists(ref_f)) else None,
                ref_audio_path_male=ref_m if (ref_m and os.path.exists(ref_m)) else None
            )

            def on_done(path):
                item.audio_path = path
                item.status = "Cloned"
                if hasattr(self, "table_subtitles") and self.table_subtitles.rowCount() > row_idx:
                    stat_it = self.table_subtitles.item(row_idx, 5)
                    if stat_it:
                        stat_it.setText("Cloned")
                        stat_it.setForeground(QColor("#10b981"))
                    self.table_subtitles.setCellWidget(row_idx, 6, self._create_row_action_widget(row_idx, item))
                self.lbl_status.setText(f"Clone បន្ទាត់ទី {row_idx + 1} បានជោគជ័យ!")

            self.worker.cloning_finished.connect(on_done)
            self.worker.error_occurred.connect(self.on_clone_error)
            self.worker.start()

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() == 3:
            row = item.row()
            if 0 <= row < len(self.subtitles):
                self.subtitles[row].translated_text = item.text()

    def on_worker_row_finished(self, row_idx: int, audio_path: str):
        if hasattr(self, "table_subtitles") and self.table_subtitles.rowCount() > row_idx:
            stat_it = self.table_subtitles.item(row_idx, 5)
            if stat_it:
                stat_it.setText("Cloned")
                stat_it.setForeground(QColor("#10b981"))
            if 0 <= row_idx < len(self.subtitles):
                self.table_subtitles.setCellWidget(row_idx, 6, self._create_row_action_widget(row_idx, self.subtitles[row_idx]))

    def on_start_clone(self):
        engine_mode = getattr(self, "current_engine_mode", "colab")
        if engine_mode == "edge":
            mode = "edge_tts"
        elif getattr(self, "is_standalone", False):
            mode = "colab_voxcpm"
        else:
            mode_idx = self.combo_clone_mode.currentIndex()
            if mode_idx == 0:
                mode = "voxcpm"
            elif mode_idx == 1:
                mode = "colab_voxcpm"
            elif mode_idx == 2:
                mode = "cloud_api"
            else:
                mode = "fast_neural"

        is_srt_tab = (hasattr(self, "stack_text_mode") and self.stack_text_mode.currentIndex() == 1)
        is_batch = (self.radio_batch.isChecked() and bool(self.subtitles)) or is_srt_tab
        self.is_batch_mode = is_batch

        batch_items = []
        if is_srt_tab:
            raw_srt = self.edit_srt.toPlainText().strip()
            if not raw_srt:
                QMessageBox.warning(self, "ខ្វះអត្ថបទ Subtitle", "សូមបញ្ចូលអត្ថបទទម្រង់ SRT ឬចុចប៊ូតុង 'ផ្ទុកហ្វាល់ SRT' ជាមុនសិន។")
                return
            srt_parsed = self._parse_srt_content(raw_srt)
            if not srt_parsed:
                QMessageBox.warning(self, "ទម្រង់មិនត្រឹមត្រូវ", "មិនអាចស្វែងរកទិន្នន័យ Subtitle SRT ត្រឹមត្រូវ (ឧ. 00:00:00,000 --> 00:00:02,000) សូមពិនិត្យមើលអត្ថបទឡើងវិញ។")
                return
            from types import SimpleNamespace
            for it in srt_parsed:
                s_obj = SimpleNamespace(
                    id=it["id"],
                    item_id=it["id"],
                    start_time=it["start_str"],
                    end_time=it["end_str"],
                    start_ms=it["start_ms"],
                    end_ms=it["end_ms"],
                    text=it["text"],
                    original_text=it["text"],
                    translated_text=it["text"],
                    audio_path="",
                    gender="male",
                    status="Pending"
                )
                batch_items.append((it["id"], it["text"], s_obj))
            text = batch_items[0][1]
        elif is_batch:
            for idx, s in enumerate(self.subtitles):
                if hasattr(self, "table_subtitles") and self.table_subtitles.rowCount() > idx:
                    t_item = self.table_subtitles.item(idx, 3)
                    if t_item:
                        s.translated_text = t_item.text().strip()
                t = (s.translated_text if s.translated_text else s.original_text).strip()
                if t:
                    batch_items.append((s.item_id, t, s))
            if not batch_items:
                QMessageBox.warning(self, "ខ្វះអត្ថបទ", "មិនមានអត្ថបទនៅក្នុងតារាង Subtitles ទេ។")
                return
            text = batch_items[0][1]
        else:
            text = self.edit_text.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "ខ្វះអត្ថបទ", "សូមបញ្ចូលអត្ថបទដែលត្រូវនិយាយ។")
                return

        ref_male = self.txt_ref_male.text().strip() if hasattr(self, "txt_ref_male") else ""
        ref_female = self.txt_ref_female.text().strip() if hasattr(self, "txt_ref_female") else ""
        ref_path = ref_male or ref_female
        if not ref_path and hasattr(self, "txt_ref_path"):
            ref_path = self.txt_ref_path.text().strip()

        if mode in ["voxcpm", "colab_voxcpm"]:
            if not ref_path or not os.path.exists(ref_path):
                reply = QMessageBox.question(
                    self, "ខ្វះសំឡេងគំរូ",
                    "សម្រាប់ Voice Clone សូមជ្រើសរើសឯកសារសំឡេងគំរូជាមុនសិន។\n\n"
                    "តើបងចង់ប្តូរទៅប្រើ [🗣️ ជម្រើសទី ២: សំឡេងខ្មែរ AI ពីសិទ្ធិ / ស្រីមុំ] ជំនួសវិញដែរឬទេ? (មិនចាំបាច់មានសំឡេងគំរូ និងមិនបាច់ Colab ឡើយ)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._set_engine_mode("edge")
                    self.on_start_clone()
                return

        colab_url = self.txt_colab_url.text().strip() if hasattr(self, "txt_colab_url") else ""
        if mode == "colab_voxcpm":
            if not colab_url:
                reply = QMessageBox.question(
                    self, "ខ្វះ Colab URL",
                    "មិនទាន់មាន Link Google Colab Server ទេ!\n\n"
                    "តើបងចង់ប្តូរទៅប្រើ [🗣️ ជម្រើសទី ២: សំឡេងខ្មែរ AI ពីសិទ្ធិ / ស្រីមុំ] ជំនួសវិញដែរឬទេ? (ដំណើរការភ្លាមៗ ឥតគិតថ្លៃ)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._set_engine_mode("edge")
                    self.on_start_clone()
                return

        prov_idx = self.combo_api_provider.currentIndex()
        if prov_idx == 0:
            api_prov = "elevenlabs"
        elif prov_idx == 1:
            api_prov = "kiritts"
        else:
            api_prov = "custom"
        api_k = self.txt_api_key.text().strip()
        api_ep = self.txt_endpoint.text().strip()

        # Extract selected voice ID
        if mode == "edge_tts":
            v_id = getattr(self, "selected_edge_voice", "km-KH-PisethNeural")
        else:
            v_data = self.combo_voice_id.itemData(self.combo_voice_id.currentIndex())
            raw_v = self.combo_voice_id.currentText().strip()
            v_id = str(v_data if v_data is not None else raw_v)

        # Emotion & Speed controls
        speed_val = float(self.combo_speed.currentData() or 1.0)
        emotion_val = str(self.combo_emotion.currentData() or "normal")
        style_prompt_val = self.txt_style_prompt.text().strip()

        if v_id == "Arun_villain":
            v_id = "Arun"
            emotion_val = "angry"
        elif v_id == "Arun_elder":
            v_id = "Arun"
            speed_val = 0.85
        elif v_id == "Chanda_young":
            v_id = "Chanda"
            emotion_val = "gentle"
        elif "Auto Clone" in v_id:
            v_id = ""
        elif v_id.startswith("🎙️"):
            v_id = v_id.replace("🎙️", "").strip()

        if mode == "cloud_api":
            if not api_k:
                QMessageBox.warning(self, "ខ្វះ API Key", "សូមបញ្ចូល API Key មុននឹងចាប់ផ្តើម Clone តាមរយៈ Cloud API។")
                return
            if v_id and v_id not in ["Maly", "Bora", "Chanda", "Arun", "__custom__"]:
                self._save_custom_voice_to_history(prov_idx, v_id)
            if not v_id and (not ref_path or not os.path.exists(ref_path)):
                QMessageBox.warning(self, "ខ្វះសំឡេង", "សូមជ្រើសរើសសំឡេង (Voice) ឬជ្រើសរើសឯកសារសំឡេងគំរូ (Reference Audio)។")
                return

        steps_map = {0: 5, 1: 10, 2: 15, 3: 20}
        cfg_map = {0: 1.5, 1: 2.0, 2: 2.5}
        timesteps = steps_map.get(self.combo_steps.currentIndex(), 10)
        cfg_val = cfg_map.get(self.combo_cfg.currentIndex(), 2.0)
        model_path = self.txt_model.text().strip() or "openbmb/VoxCPM2"
        prompt_txt = self.txt_prompt_text.text().strip() or None

        device_idx = self.combo_device.currentIndex()
        device_str = "auto" if device_idx == 0 else ("cpu" if device_idx == 1 else "cuda")

        # Release and stop player_res to avoid Windows file locking errors
        try:
            self.player_res.stop()
            self.player_res.setSource(QUrl())
            self.btn_play_result.setText("▶ ចាក់ស្តាប់សំឡេង Clone")
        except Exception:
            pass

        import time
        out_wav = str((OUTPUT_DIR / f"cloned_voice_{int(time.time())}.wav").resolve())

        merge_mode_val = "continuous"
        trim_silence_val = True
        if is_srt_tab:
            if hasattr(self, "combo_srt_dub_mode"):
                merge_mode_val = str(self.combo_srt_dub_mode.currentData() or "continuous")
            if hasattr(self, "chk_srt_trim_silence"):
                trim_silence_val = bool(self.chk_srt_trim_silence.isChecked())

        self.btn_start_clone.setEnabled(False)
        self.btn_play_result.setEnabled(False)
        self.btn_save_audio.setEnabled(False)
        self.progress_bar.setValue(10)
        self.lbl_status.setText("កំពុងដំណើរការ...")

        self.worker = VoiceCloneWorker(
            text=text,
            ref_audio_path=ref_path,
            output_wav_path=out_wav,
            mode=mode,
            prompt_text=prompt_txt,
            model_path=model_path,
            cfg_value=cfg_val,
            timesteps=timesteps,
            device=device_str,
            api_provider=api_prov,
            api_key=api_k,
            api_endpoint=api_ep,
            voice_id=v_id,
            speed=speed_val,
            emotion=emotion_val,
            style_prompt=style_prompt_val,
            batch_items=batch_items,
            colab_url=colab_url,
            ref_audio_path_female=ref_female if (ref_female and os.path.exists(ref_female)) else None,
            ref_audio_path_male=ref_male if (ref_male and os.path.exists(ref_male)) else None,
            merge_mode=merge_mode_val,
            trim_leading_silence=trim_silence_val
        )
        self.worker.status_updated.connect(self.lbl_status.setText)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.row_finished.connect(self.on_worker_row_finished)
        self.worker.cloning_finished.connect(self.on_clone_completed)
        self.worker.error_occurred.connect(self.on_clone_error)
        self.worker.start()

    def on_clone_completed(self, out_audio_path):
        self.btn_start_clone.setEnabled(True)
        self.output_cloned_wav = out_audio_path
        self.btn_play_result.setEnabled(True)
        self.btn_save_audio.setEnabled(True)
        self.btn_apply_table.setEnabled(True)
        if self.current_video_path and os.path.exists(self.current_video_path):
            self.btn_fast_mux.setEnabled(True)
        self.cloned_audio_ready.emit(out_audio_path)
        sub_info = f"\nសរុប: {len(self.subtitles)} ឃ្លាក្នុងតារាង Subtitles" if getattr(self, "is_batch_mode", False) else ""
        QMessageBox.information(
            self, "Clone ជោគជ័យ",
            f"🎉 សំឡេងត្រូវបាន Clone ជោគជ័យជាស្ថាពរ!{sub_info}\nឯកសារ: {out_audio_path}\nSample Rate: 48,000Hz (Studio Quality)"
        )

    def on_clone_error(self, err_msg):
        self.btn_start_clone.setEnabled(True)
        self.progress_bar.setValue(0)
        short_err = (err_msg[:60] + "...") if len(err_msg) > 60 else err_msg
        self.lbl_status.setText(f"មានបញ្ហា: {short_err}")
        if hasattr(self, "chk_limit_fallback_ai") and self.chk_limit_fallback_ai.isChecked():
            fallback_v = "km-KH-PisethNeural" if self.combo_fallback_voice.currentIndex() == 0 else "km-KH-SreymomNeural"
            v_name = "ពីសិទ្ធិ (Piseth)" if "Piseth" in fallback_v else "ស្រីមុំ (Sreymom)"
            reply = QMessageBox.question(
                self, "Google Colab ជួបបញ្ហា / Limit",
                f"Google Colab Server ជួបបញ្ហា៖\n{err_msg[:200]}\n\n"
                f"តើបងចង់ឱ្យប្រព័ន្ធបង្កើតសំឡេងដោយប្រើ [🗣️ សំឡេង AI {v_name}] ជំនួសវិញភ្លាមៗដែរឬទេ? (ឥតគិតថ្លៃ និងដំណើរការបានភ្លាមៗ)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.selected_edge_voice = fallback_v
                self._set_engine_mode("edge")
                self.on_start_clone()
                return

        QMessageBox.critical(
            self, "មានបញ្ហាក្នុងការ Clone",
            f"មិនអាច Clone សំឡេងបានទេ:\n{err_msg}\n\nអាចពិនិត្យមើល Colab Server ឬប្តូរទៅប្រើ 'ជម្រើសទី ២: សំឡេងខ្មែរ AI (ពីសិទ្ធិ/ស្រីមុំ)'!"
        )

    def on_save_cloned_audio(self):
        """Saves or exports the cloned audio file (.wav or .mp3) to a user-specified path."""
        if not self.output_cloned_wav or not os.path.exists(self.output_cloned_wav):
            QMessageBox.warning(self, "គ្មានសំឡេង", "មិនទាន់មានឯកសារសំឡេង Clone ទេ។ សូមធ្វើការ Clone សំឡេងជាមុនសិន។")
            return

        src_path = Path(self.output_cloned_wav)
        default_filename = f"Cloned_Voice_{int(time.time())}.wav"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "រក្សាទុកឯកសារសំឡេង Clone (Save Cloned Audio)",
            str(Path.home() / "Desktop" / default_filename),
            "WAV Audio (*.wav);;MP3 Audio (*.mp3);;All Files (*.*)"
        )

        if not save_path:
            return

        dest_file = Path(save_path)
        # Add extension if user did not include it
        if not dest_file.suffix:
            if "mp3" in selected_filter.lower():
                dest_file = dest_file.with_suffix(".mp3")
            else:
                dest_file = dest_file.with_suffix(".wav")

        try:
            if dest_file.suffix.lower() == ".mp3" and src_path.suffix.lower() != ".mp3":
                # Convert WAV to high-quality MP3 using ffmpeg
                cmd = [
                    FFMPEG_EXE, "-y",
                    "-i", str(src_path),
                    "-b:a", "256k",
                    str(dest_file)
                ]
                silent_subprocess_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            else:
                shutil.copy2(str(src_path), str(dest_file))

            QMessageBox.information(
                self,
                "រក្សាទុកជោគជ័យ",
                f"🎉 បានរក្សាទុកឯកសារសំឡេង Clone ដោយជោគជ័យ!\n\n📁 ទីតាំង: {dest_file}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "រក្សាទុកបរាជ័យ",
                f"មិនអាចរក្សាទុកឯកសារសំឡេងបានទេ:\n{str(e)}"
            )

    def on_apply_to_table(self):
        """Emits audio path so parent window updates subtitle row or full dub track."""
        if self.output_cloned_wav:
            self.cloned_audio_ready.emit(self.output_cloned_wav)
            QMessageBox.information(
                self, "បញ្ចូលជោគជ័យ",
                "បានកំណត់សំឡេង Cloned នេះចូលទៅក្នុង Subtitle និង Dubbed Soundtrack រួចរាល់!"
            )

    def on_fast_mux(self):
        """Fast merges cloned voice with loaded video using -c:v copy."""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(self, "គ្មានវីដេអូ", "សូមប្រាកដថាបាន Load វីដេអូជាមុនសិន។")
            return
        if not self.output_cloned_wav or not os.path.exists(self.output_cloned_wav):
            QMessageBox.warning(self, "គ្មានសំឡេង", "មិនទាន់មានសំឡេង Clone ទេ។")
            return

        v_name = Path(self.current_video_path).stem
        out_vid = f"output/{v_name}_VoxCPM_Cloned.mp4"
        try:
            mux_cloned_voice_to_video_fast(
                self.current_video_path,
                self.output_cloned_wav,
                out_vid
            )
            QMessageBox.information(
                self, "ផ្គុំវីដេអូជោគជ័យ",
                f"⚡ ផ្គុំសំឡេង Cloned ចូលវីដេអូបានសម្រេចយ៉ាងរហ័ស (-c:v copy)!\n\nវីដេអូ: {out_vid}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Mux Error", str(e))
