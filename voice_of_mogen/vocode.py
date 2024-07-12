import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import numpy as np
import pedalboard
import psola
import requests
from banana_dev import Client as BananaClient
from pedalboard import Chain
from pedalboard import Mix
from pedalboard import Pedalboard
from scipy import signal

from voice_of_mogen import Composition
from voice_of_mogen.audio import Audio

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

A4_FREQ = 440.0
A4_MIDI = 69
NOTES_PER_OCTAVE = 12

DEFAULT_VOCODER_PARAMETERS = {
    "fmin": 20.0,
    "fmax": 2000.0,
    "width": 0.8,
    "smoothing_N": 100,
    "silence_threshold": 1e-5,
    "normalize": True,
}

DEFAULT_EFFECTS_CHAIN = {
    "type": "pedalboard",
    "plugins": [
        {
            "type": "effect",
            "name": "HighpassFilter",
            "params": {"cutoff_frequency_hz": 40},
        },
        {
            "type": "effect",
            "name": "LowShelfFilter",
            "params": {"cutoff_frequency_hz": 180, "gain_db": 6.0},
        },
        {
            "type": "effect",
            "name": "PeakFilter",
            "params": {"cutoff_frequency_hz": 260, "q": 0.65, "gain_db": -5.0},
        },
        {
            "type": "effect",
            "name": "PeakFilter",
            "params": {"cutoff_frequency_hz": 1100, "q": 0.7, "gain_db": -4.0},
        },
        {
            "type": "effect",
            "name": "HighShelfFilter",
            "params": {"cutoff_frequency_hz": 4000, "gain_db": 4.0},
        },
        {
            "type": "effect",
            "name": "Chorus",
            "params": {
                "rate_hz": 0.8,
                "depth": 0.1,
                "centre_delay_ms": 6.0,
                "feedback": 0.001,
                "mix": 0.3,
            },
        },
        {
            "type": "mix",
            "plugins": [
                {
                    "type": "pedalboard",
                    "plugins": [
                        {
                            "type": "effect",
                            "name": "Reverb",
                            "params": {
                                "room_size": 0.65,
                                "damping": 0.20,
                                "wet_level": 0.04,
                                "dry_level": 0.0,
                                "width": 1.0,
                            },
                        },
                        {
                            "type": "effect",
                            "name": "HighpassFilter",
                            "params": {"cutoff_frequency_hz": 200},
                        },
                        {
                            "type": "effect",
                            "name": "LowpassFilter",
                            "params": {"cutoff_frequency_hz": 2500},
                        },
                        {
                            "type": "effect",
                            "name": "PeakFilter",
                            "params": {
                                "cutoff_frequency_hz": 800,
                                "q": 0.5,
                                "gain_db": -6.0,
                            },
                        },
                    ],
                },
                {"type": "effect", "name": "Gain", "params": {"gain_db": 0}},
            ],
        },
        {
            "type": "effect",
            "name": "Compressor",
            "params": {
                "threshold_db": -12.0,
                "ratio": 3.5,
                "attack_ms": 13.0,
                "release_ms": 105.0,
            },
        },
    ],
}
DEFAULT_SPEECH_EFFECTS_CHAIN = [{"type": "pedalboard", "plugins": []}]


def _effect_has_children(effect_node: dict) -> bool:
    return "plugins" in effect_node


def _construct_effect(effect_node: dict) -> Pedalboard:
    if _effect_has_children(effect_node):
        # print("effect has children", effect_node)
        children = [_construct_effect(child) for child in effect_node["plugins"]]

        if effect_node["type"] == "pedalboard":
            return Pedalboard(children)
        elif effect_node["type"] == "chain":
            return Chain(children)
        elif effect_node["type"] == "mix":
            return Mix(children)
        else:
            raise ValueError(f"Unknown chain type {effect_node['type']}")
    elif isinstance(effect_node, dict) and effect_node["type"] == "effect":
        # print("effect is effect", effect_node)
        effect_cls = getattr(pedalboard, effect_node["name"])

        if (
            effect_node["name"] == "LadderFilter"
            and type(effect_node["params"]["mode"]) == str
        ):
            effect_node["params"]["mode"] = getattr(
                pedalboard.LadderFilter.Mode, effect_node["params"]["mode"]
            )

        return effect_cls(**effect_node["params"])
    elif isinstance(effect_node, dict):
        raise ValueError(f"Unknown node type {effect_node['type']}")
    else:
        return Pedalboard([])


def make_effects_chain(effect_spec: dict) -> Pedalboard:
    return _construct_effect(effect_spec)


def mtof(midi: float) -> float:
    return A4_FREQ * 2 ** ((midi - A4_MIDI) / NOTES_PER_OCTAVE)


class IVocoder(ABC):
    @abstractmethod
    def process(
        self, audio: Audio, composition: Composition, synthesis_parameters: dict
    ) -> Audio:
        """
        Process the audio with the given composition.

        Args:
            audio: The audio to process.
            composition: The composition to use.
            synthesis_parameters: The parameters to use for audio synthesis.

        Returns:
            A tuple of the processed audio and the start index of the tail of the
            effects chain.
        """
        pass


class APIVocoder(IVocoder):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def process(
        self, audio: Audio, composition: Composition, synthesis_parameters: dict
    ) -> Audio:
        encoded_audio = audio.encode()
        encoded_composition = composition.to_dict()
        request = {
            "speech_audio": encoded_audio,
            "composition": encoded_composition,
            "synthesis_parameters": synthesis_parameters,
        }

        response = requests.post(self.endpoint_url, json=request)
        response.raise_for_status()

        decoded_audio = Audio.decode(response.json()["audio"])
        decoded_tail_idx = response.json()["tail_idx"]
        return Audio(
            audio_data=decoded_audio,
            sample_rate=audio.sample_rate,
            duration_in_seconds=audio.duration_in_seconds,
            metadata=dict(tail_start_in_samples=decoded_tail_idx),
        )


class BananaVocoder(IVocoder):
    def __init__(self, endpoint_url: str, api_key: str, model_key: str):
        self.client = BananaClient(
            model_key=model_key,
            api_key=api_key,
            url=endpoint_url,
        )

    def process(
        self, audio: Audio, composition: Composition, synthesis_parameters: dict
    ) -> Audio:
        assert audio.is_remote, "Audio must be remote"

        encoded_composition = composition.to_dict()
        request = {
            "speech_audio_url": audio.clip_url,
            "sample_rate": audio.sample_rate,
            "composition": encoded_composition,
            "synthesis_parameters": synthesis_parameters,
        }

        response, _ = self.client.call("/vocode", request)

        return Audio(
            clip_url=response["audio_url"],
            sample_rate=audio.sample_rate,
            metadata=dict(
                tail_start_in_samples=response["tail_start_in_samples"],
            ),
        )


class PSOLAVocoder(IVocoder):
    def __init__(
        self,
        effects_buffer_size: int = 8192,
    ):
        self.effects_buffer_size = effects_buffer_size

    def _apply_fx(
        self,
        audio: np.ndarray,
        sample_rate: int,
        silence_threshold: float,
        effect_spec: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        effects_chain = make_effects_chain(effect_spec)
        
        try:
            # Process the audio using the PedalBoard instance
            output = effects_chain(audio, sample_rate, buffer_size=self.effects_buffer_size)
            print("Audio processing completed successfully.")
        except Exception as e:
            print(f"An error occurred during audio processing: {e}")

        buffers = []
        silence = np.zeros((output.shape[0], self.effects_buffer_size))
        while True:
            buffer = effects_chain(
                silence,
                sample_rate,
                buffer_size=self.effects_buffer_size,
                reset=False,
            )
            if np.max(np.abs(buffer)) < silence_threshold:
                break

            buffers.append(buffer)

        if len(buffers) == 0:
            tail = np.zeros((output.shape[0], 0))
        else:
            tail = np.concatenate(buffers, axis=1)
        return output, tail

    def process(
        self, audio: Audio, composition: Composition, synthesis_parameters: dict
    ) -> Audio:
        voice_outputs = []

        if (
            synthesis_parameters is not None
            and "vocoder_parameters" in synthesis_parameters
            and synthesis_parameters["vocoder_parameters"] is not None
            and synthesis_parameters["vocoder_parameters"] != {}
        ):
            vocoder_params = synthesis_parameters["vocoder_parameters"]
        else:
            vocoder_params = DEFAULT_VOCODER_PARAMETERS

        smoothing_N = vocoder_params["smoothing_N"]

        log.info(
            f"({self.__class__.__name__}) Processing {composition.num_voices} voices"
        )
        for i in range(composition.num_voices):
            log.debug(f"({self.__class__.__name__}) Processing voice {i}")
            log.debug(f"({self.__class__.__name__}) Creating frequency track")
            pitch_track = composition.notes_as_time_series(i, audio.sample_rate)
            pitch_track = np.pad(
                pitch_track,
                (0, max(audio.audio_data.shape[-1] - pitch_track.shape[0], 0)),
            )
            pitch_track = pitch_track[: audio.audio_data.shape[-1]]

            active = (pitch_track > 0).astype(float)
            freq_track = mtof(pitch_track)

            log.debug(
                f"({self.__class__.__name__}) Smoothing activity and freq tracks..."
            )
            # smooth the activity and frequency tracks with a moving average filter
            active = signal.convolve(
                active, np.ones(smoothing_N) / smoothing_N, mode="same"
            )
            freq_track = signal.convolve(
                freq_track, np.ones(smoothing_N) / smoothing_N, mode="same"
            )

            log.debug(f"({self.__class__.__name__}) Vocoding...")
            voice = psola.vocode(
                audio.audio_data,
                audio.sample_rate,
                target_pitch=freq_track,
                fmin=vocoder_params["fmin"],
                fmax=vocoder_params["fmax"],
            )

            log.debug(f"({self.__class__.__name__}) Applying activity mask...")
            voice = voice * active.astype(float)
            voice_outputs.append(voice)

        output = np.stack(voice_outputs, axis=0)
        output = self._pan_voices(output, vocoder_params["width"])
        output = output.mean(axis=0)[:, : audio.audio_data.shape[-1]]

        if (
            synthesis_parameters is not None
            and "vocoder_effect_spec" in synthesis_parameters
        ):
            vocoder_effect_spec = synthesis_parameters["vocoder_effect_spec"]
        else:
            vocoder_effect_spec = DEFAULT_EFFECTS_CHAIN

        if (
            synthesis_parameters is not None
            and "speech_effect_spec" in synthesis_parameters
        ):
            speech_effect_spec = synthesis_parameters["speech_effect_spec"]
        else:
            speech_effect_spec = DEFAULT_SPEECH_EFFECTS_CHAIN

        if vocoder_params["normalize"]:
            output = output / np.clip(np.max(np.abs(output)), 1e-5, None)
        vocoder_output, vocoder_tail = (
            self._apply_fx(
                output,
                audio.sample_rate,
                vocoder_params["silence_threshold"],
                vocoder_effect_spec,
            )
            if vocoder_effect_spec is not None
            else (np.zeros_like(output), np.zeros((output.shape[0], 0)))
        )
        tail_idx = output.shape[-1]
        vocoder_output = (
            np.concatenate((vocoder_output, vocoder_tail), axis=1)
            if vocoder_tail.shape[-1] > 0
            else vocoder_output
        )
        print("vocoder_output", vocoder_output.shape)

        speech_output, speech_tail = (
            self._apply_fx(
                audio.audio_data[None],
                audio.sample_rate,
                vocoder_params["silence_threshold"],
                speech_effect_spec,
            )
            if speech_effect_spec is not None
            else (
                np.zeros_like(audio.audio_data),
                np.zeros((audio.audio_data.shape[0], 0)),
            )
        )
        tail_idx = max(tail_idx, speech_output.shape[-1])
        speech_output = (
            np.concatenate((speech_output, speech_tail), axis=1)
            if speech_tail.shape[-1] > 0
            else speech_output
        )
        print("speech_output", speech_output.shape)

        # pad the shorter output to match the longer one
        if vocoder_output.shape[-1] < speech_output.shape[-1]:
            vocoder_output = np.pad(
                vocoder_output,
                ((0, 0), (0, speech_output.shape[-1] - vocoder_output.shape[-1])),
            )
        elif speech_output.shape[-1] < vocoder_output.shape[-1]:
            speech_output = np.pad(
                speech_output,
                ((0, 0), (0, vocoder_output.shape[-1] - speech_output.shape[-1])),
            )

        scale = (
            0.5
            if speech_effect_spec is not None and vocoder_effect_spec is not None
            else 1.0
        )
        output = scale * (vocoder_output + speech_output)

        return Audio(
            audio_data=output,
            sample_rate=audio.sample_rate,
            metadata=dict(tail_start_in_samples=tail_idx),
        )

    def _pan_voices(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Pans the voices in the audio array to the left and right channels.

        Args:
            audio (np.ndarray): Audio array with shape (num_voices, num_samples)

        Returns:
            np.ndarray: Audio array with shape (2, num_samples)
        """
        left = audio * np.linspace(1, 1 - width, audio.shape[0])[:, None]
        right = audio * np.linspace(1 - width, 1, audio.shape[0])[:, None]
        return np.stack((left, right), axis=1)
