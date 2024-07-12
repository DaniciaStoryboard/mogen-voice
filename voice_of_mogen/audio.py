import base64
import logging
from io import BytesIO
from typing import Literal
from typing import Optional

import numpy as np
import requests
import soundfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Audio:
    """Object representing an audio signal and providing helper methods for
    manipulating, storing, and fetching it.
    """

    audio_buffer: Optional[np.ndarray]  # (channels, samples)
    sample_rate: Optional[int]
    clip_url: Optional[str]
    duration_in_seconds: Optional[float]
    original_source: Literal["remote", "local"]
    uploaded: bool = False
    metadata: dict = {}

    def __init__(
        self,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = 44100,
        clip_url: Optional[str] = None,
        duration_in_seconds: Optional[float] = None,
        metadata: Optional[dict] = None,
        normalize: bool = False,
    ):
        if audio_data is not None:
            assert (
                clip_url is None
            ), "`clip_url` must be None if `audio_data` is provided"

            assert (
                sample_rate is not None
            ), "`sample_rate` must be provided if `audio_data` is provided"
            if duration_in_seconds is None:
                duration_in_seconds = audio_data.shape[-1] / sample_rate
            self.original_source = "local"
        elif clip_url is not None:
            self.original_source = "remote"

        assert (
            clip_url is not None or audio_data is not None
        ), "Either `clip_url` or `audio_data` must be provided"

        self.audio_buffer = audio_data
        self.sample_rate = sample_rate
        self.clip_url = clip_url
        self.duration_in_seconds = duration_in_seconds
        self.metadata = metadata or {}
        self.normalize = normalize

    @property
    def is_local(self) -> bool:
        return self.original_source == "local"

    @property
    def is_remote(self) -> bool:
        return self.original_source == "remote"

    @property
    def is_fetched(self) -> bool:
        return self.is_remote and self.audio_buffer is not None

    @property
    def is_uploaded(self) -> bool:
        return self.is_local and self.uploaded

    @property
    def audio_data(self) -> np.ndarray:
        if self.is_remote and not self.is_fetched:
            self.fetch()
        return self.audio_buffer

    def fetch(self) -> None:
        """Fetch audio data from remote URL."""
        log.info(f"Fetching audio from {self.clip_url}...")
        audio_response = requests.get(self.clip_url)
        audio_response.raise_for_status()
        log.info("Done.")
        bytes = audio_response.content
        audio, sr = soundfile.read(BytesIO(bytes))

        if self.sample_rate is not None:
            ## I modified this part
            assert sr == int(self.sample_rate), "Sample rate mismatch"

        self.audio_buffer = audio
        self.sample_rate = sr

        if self.normalize:
            self.audio_buffer = self.audio_buffer / np.clip(
                np.max(np.abs(self.audio_buffer)), 1e-5, None
            )

    def encode(self) -> dict:
        """Encode audio data as a dictionary for JSON serialization."""
        if self.is_remote:
            assert (
                self.is_fetched
            ), "Audio data must be fetched before encoding for remote audio"

        audio_bytes = self.audio_buffer.astype(np.float32).tobytes()
        audio_string = base64.b64encode(audio_bytes).decode("utf-8")
        return dict(
            audio_str=audio_string,
            audio_shape=self.audio_buffer.shape,
            sample_rate=self.sample_rate,
        )

    @classmethod
    def decode(cls, encoded_audio: dict) -> "Audio":
        """Decode audio data from a dictionary."""
        audio_bytes = base64.b64decode(encoded_audio["audio_str"])
        audio = np.frombuffer(audio_bytes, dtype=np.float32).reshape(
            encoded_audio["audio_shape"]
        )
        return cls(audio_data=audio, sample_rate=encoded_audio["sample_rate"])
