import base64
import os
import sys
import shutil
import logging
import tempfile
import warnings

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, whoami
import torch
import numpy as np
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
from speaker_processing import (
    process_diarized_output, enroll_profiles, identify_speakers_on_segments,
    load_known_speakers_from_samples, identify_speaker,
    free_embed_model, free_ecapa_model,
    _SPEAKER_EMBEDDING_CACHE,
)

# ---------------------------------------------------------------------------
# Performance: enable TF32 for faster matmul on Ampere+ GPUs (L40S, A100, H100)
# Safe for transcription — negligible accuracy impact, significant speed gain
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Suppress known harmless warnings that spam every job
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends has been deprecated.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*callback states.*colliding.*")
warnings.filterwarnings("ignore", message=".*Model has been trained with a task-dependent loss function.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", message=".*Redirecting import of pytorch_lightning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_' prefix - token may be malformed")

if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")




MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # Clear speaker embedding cache from previous jobs
    _SPEAKER_EMBEDDING_CACHE.clear()

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        # Flatten error list to string — RunPod SDK can't serialize list values
        errors = validated["errors"]
        if isinstance(errors, (list, dict)):
            errors = str(errors)
        return {"error": errors}

    # ------------- 1) resolve audio input (URL or base64) -----------
    audio_input = job_input["audio_file"]
    try:
        if "://" in audio_input:
            # Standard URL — download as before
            audio_file_path = download_files_from_urls(job_id, [audio_input])[0]
            logger.debug(f"Audio downloaded → {audio_file_path}")
        else:
            # Treat as base64-encoded audio data
            # Strip optional data-URI prefix (e.g. "data:audio/wav;base64,")
            if "," in audio_input:
                audio_input = audio_input.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_input)
            os.makedirs(f"/jobs/{job_id}", exist_ok=True)
            audio_file_path = f"/jobs/{job_id}/audio_input.wav"
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            logger.debug(f"Audio decoded from base64 → {audio_file_path} ({len(audio_bytes)} bytes)")
    except Exception as e:
        logger.error("Audio input failed", exc_info=True)
        return {"error": f"audio input: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token  # or job_input.get("huggingface_access_token")
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            embeddings = {}  # graceful degradation: proceed without profiles

    # Free embedding model from VRAM before loading Whisper.
    # Enrolled embeddings are numpy arrays on CPU — safe to free the GPU model.
    # It will lazy-reload in step 4 (speaker verification) after Whisper is freed.
    if speaker_profiles:
        free_embed_model()
        free_ecapa_model()

    # ------------- 3) call WhisperX / VAD / diarization -------------
    _prompt = job_input.get("initial_prompt")
    _hotwords = job_input.get("hotwords")
    if _prompt or _hotwords:
        logger.info(f"Vocab hints: initial_prompt={repr(_prompt[:80] + '...' if _prompt and len(_prompt) > 80 else _prompt)}, "
                     f"hotwords={repr(_hotwords[:80] + '...' if _hotwords and len(_hotwords) > 80 else _hotwords)}")

    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : job_input.get("huggingface_access_token") or hf_token,
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
        "hotwords"                 : job_input.get("hotwords"),
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language
    }
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    if embeddings:
        try:
            output_dict["segments"], output_dict["speaker_map"] = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.55
            )
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # Free speaker models from VRAM after verification is done
    free_embed_model()
    free_ecapa_model()

    # 4-Cleanup and return output_dict normally
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict

runpod.serverless.start({"handler": run})