import base64
import json
import logging
import os
import secrets
import string
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import soundfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def decode_audio(encoded_audio: dict) -> np.ndarray:
    audio_bytes = base64.b64decode(encoded_audio["audio_str"])
    audio = np.frombuffer(audio_bytes, dtype=np.float32).reshape(
        encoded_audio["audio_shape"]
    )
    return audio


def encode_audio(audio: np.ndarray) -> dict:
    audio_bytes = audio.astype(np.float32).tobytes()
    audio_str = base64.b64encode(audio_bytes).decode("utf-8")
    return dict(audio_str=audio_str, audio_shape=audio.shape)


def get_env():
    config = {
        "CACHE_BUCKET": os.environ.get("CACHE_BUCKET", None),
        "CHECKPOINT_BUCKET": os.environ.get("CHECKPOINT_BUCKET", None),
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON", None
        ),
        "GOOGLE_PROJECT": os.environ.get("GOOGLE_PROJECT", None),
        "GOOGLE_SERVICE_ACCOUNT": os.environ.get("GOOGLE_SERVICE_ACCOUNT", None),
        "PREFIX_COMPOSITION": os.environ.get("PREFIX_COMPOSITION", None),
        "PREFIX_SENTIMENT": os.environ.get("PREFIX_SENTIMENT", None),
        "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", None
        ),
    }

    for k, v in config.items():
        if v is None:
            raise Exception(f"{k} is not set")

    return config


def set_adc(config):
    adc_path = config["GOOGLE_APPLICATION_CREDENTIALS"]
    adc_credentials = config["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

    # create directory if not exists
    Path(adc_path).parent.mkdir(parents=True, exist_ok=True)

    # write credentials
    with open(adc_path, "w") as f:
        json.dump(json.loads(adc_credentials), f)


def download_gcs_folder(bucket, folder, output_dir):
    blobs = bucket.list_blobs(prefix=folder)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        file_name = blob.name.split("/")[-1]
        file_path = Path(output_dir) / folder / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file_name}...")
        blob.download_to_filename(file_path)


def upload_folder_to_gcs(bucket, folder, input_dir):
    for file_name in os.listdir(input_dir):
        file_path = Path(input_dir) / file_name
        blob = bucket.blob(f"{folder}/{file_name}")
        blob.upload_from_filename(file_path)


def make_fake_name(num_chars: int = 16, prefix: str = ""):
    if prefix != "":
        prefix = prefix + "-"
    return prefix + (
        "".join(
            secrets.choice(string.ascii_lowercase + string.digits)
            for _ in range(num_chars)
        )
        + ".wav"
    )


def upload_audio_to_gcs(bucket, file_name, audio, sr):
    wav_io = BytesIO()

    soundfile.write(wav_io, audio.T, sr, format="WAV")

    blob = bucket.blob(file_name)
    blob.upload_from_string(wav_io.getvalue(), content_type="audio/wav")
    blob.make_public()

    return blob.public_url


def list_bucket_contents(bucket):
    return [blob.name for blob in bucket.list_blobs()]


def fetch_audio(audio_url):
    log.info(f"Fetching audio from {audio_url}...")
    audio_response = requests.get(audio_url)
    audio_response.raise_for_status()
    log.info("Done.")
    bytes = audio_response.content
    audio, sr = soundfile.read(BytesIO(bytes))
    return audio, sr
