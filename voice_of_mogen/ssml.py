"""
Augment text with SSML tags for better TTS synthesis.
In future we will add LLM-based text augmentation, but for now we simply wrap the full
text in SSML tags.
"""
import logging
from typing import Literal
from typing import Optional

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _make_prosody_tag(
    pitch: Optional[Literal["x-low", "low", "medium", "high", "x-high"]] = None,
    rate: Optional[float] = None,
    volume: Optional[
        Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"]
    ] = None,
    *args,
    **kwargs,
) -> str:
    pitch_attr = f'pitch="{pitch}"' if pitch is not None else ""
    rate_attr = f'rate="{rate}"' if rate is not None else ""
    volume_attr = f'volume="{volume}"' if volume is not None else ""

    prosody_open_tag = f"<prosody {pitch_attr} {rate_attr} {volume_attr}>"
    prosody_close_tag = "</prosody>"

    return prosody_open_tag, prosody_close_tag


def _make_resemble_emotion_tag(
    emotion: Optional[
        Literal["neutral", "angry", "annoyed", "question", "happy"]
    ] = None,
    pitch: Optional[float] = None,
    intensity: Optional[float] = None,
    pace: Optional[float] = None,
    *args,
    **kwargs,
) -> str:
    emotion_attr = f'emotions="{emotion}"' if emotion is not None else ""
    pitch_attr = f'pitch="{pitch}"' if pitch is not None else ""
    intensity_attr = f'intensity="{intensity}"' if intensity is not None else ""
    pace_attr = f'pace="{pace}"' if pace is not None else ""

    resemble_emotion_open_tag = (
        f"<resemble:emotion {emotion_attr} {pitch_attr} {intensity_attr} {pace_attr}>"
    )
    resemble_emotion_close_tag = "</resemble:emotion>"

    return resemble_emotion_open_tag, resemble_emotion_close_tag


def _add_pauses(
    text: str,
    long_pause_duration: float = 0.5,
    short_pause_duration: float = 0.3,
    long_pause_characters: str = ".!?—–",
    short_pause_characters: str = ",;:",
    *args,
    **kwargs,
) -> str:
    for char in long_pause_characters:
        text = text.replace(
            f"{char} ", f"{char}<break time='{long_pause_duration}s'/> "
        )
    for char in short_pause_characters:
        text = text.replace(
            f"{char} ", f"{char}<break time='{short_pause_duration}s'/> "
        )

    return text


def augment_text(
    text: str,
    prosody_settings: Optional[dict] = None,
    resemble_emotion_settings: Optional[dict] = None,
    pause_settings: Optional[dict] = None,
) -> str:
    prosody_tag_is_necessary = (
        prosody_settings is not None
        and "use_prosody" in prosody_settings
        and prosody_settings["use_prosody"]
    )
    resemble_emotion_tag_is_necessary = (
        resemble_emotion_settings is not None
        and "use_emotion" in resemble_emotion_settings
        and resemble_emotion_settings["use_emotion"]
    )
    pause_augmentation_is_necessary = (
        pause_settings is not None
        and "add_pauses" in pause_settings
        and pause_settings["add_pauses"]
    )

    if pause_augmentation_is_necessary:
        text = _add_pauses(text, **pause_settings)

    if prosody_tag_is_necessary:
        prosody_open_tag, prosody_close_tag = _make_prosody_tag(**prosody_settings)
        text = f"{prosody_open_tag}{text}{prosody_close_tag}"

    if resemble_emotion_tag_is_necessary:
        (
            resemble_emotion_open_tag,
            resemble_emotion_close_tag,
        ) = _make_resemble_emotion_tag(**resemble_emotion_settings)
        text = f"{resemble_emotion_open_tag}{text}{resemble_emotion_close_tag}"

    return f"<speak>{text}</speak>"
