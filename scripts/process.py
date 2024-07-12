import os
import re
from io import BytesIO
from queue import Queue
from threading import Thread

import numpy as np
import requests
import simpleaudio as sa
import soundfile
from dotenv import load_dotenv
from resemble import Resemble

from voice_of_mogen import VoiceOfMogen

SPLIT_CHARS = ".?!"
SPLIT_GROUP = "(" + "|".join([re.escape(c) for c in SPLIT_CHARS]) + "|$)"
SPLIT_CLASS = f"[^{re.escape(SPLIT_CHARS)}]"


def init_vom():
    banana_api_key = os.getenv("BANANA_API_KEY")
    composition_endpoint = os.getenv("COMPOSITION_ENDPOINT")
    sentiment_endpoint = os.getenv("SENTIMENT_ENDPOINT")
    composition_model_key = os.getenv("COMPOSITION_MODEL_KEY")
    sentiment_model_key = os.getenv("SENTIMENT_MODEL_KEY")

    print("Initialising VoiceOfMogen...")
    voice_of_mogen = (
        VoiceOfMogen(44100)
        .with_sentiment_analyzer(
            "banana",
            endpoint_url=sentiment_endpoint,
            api_key=banana_api_key,
            model_key=sentiment_model_key,
        )
        .with_composer(
            "banana",
            endpoint_url=composition_endpoint,
            api_key=banana_api_key,
            model_key=composition_model_key,
        )
        .with_vocoder("psola")
        .init()
    )
    print("Done.")
    return voice_of_mogen


def init_audio():
    audio_queue = Queue()

    global audio_thread
    audio_thread = Thread(target=audio_consumer, args=(audio_queue,))
    audio_thread.start()

    return audio_queue


def configure_resemble():
    print("Initialising Resemble...")
    api_key = os.getenv("RESEMBLE_API_KEY")
    Resemble.api_key(api_key)
    projects_response = Resemble.v2.projects.all(1, 10)
    project_uuid = projects_response["items"][0]["uuid"]
    clips_response = Resemble.v2.clips.all(project_uuid, 1, 10)
    print("Done.")

    return clips_response, project_uuid


def fetch_clip(clips, project_uuid: str):
    print("Available clips:")
    for i, clip in enumerate(clips["items"]):
        print(f"({i}): {clip['body']}")
    clip_id = input("Select clip ID: ")
    clip_id = int(clip_id)

    clip = clips["items"][clip_id]
    audio_url = clip["audio_src"]
    print(f"Fetching audio from {audio_url}...")
    audio_response = requests.get(audio_url)
    audio_response.raise_for_status()
    print("Done.")
    bytes = audio_response.content
    audio, sr = soundfile.read(BytesIO(bytes))

    # phon_chars, phon_times
    hit_points = []
    word_active = False

    for i, (char, times) in enumerate(
        zip(clip["timestamps"]["phon_chars"], clip["timestamps"]["phon_times"])
    ):
        if not word_active and char != " ":
            word_active = True
            hit_points.append(times[0])
        elif word_active and char == " ":
            word_active = False
    hit_points.append(clip["timestamps"]["phon_times"][-1][-1])

    text = "".join(clip["timestamps"]["graph_chars"])

    return audio, hit_points, text, sr


def to_sentences(text: str) -> list[str]:
    sentences = [
        s
        for s, _ in re.findall(f"({SPLIT_CLASS}*?{SPLIT_GROUP})", text)
        if s.strip() != ""
    ]
    print(f"Split into {len(sentences)} sentences.")
    print(f"Sentences: {sentences}")
    return sentences


def to_ssml(text: str) -> str:
    words = text.split(" ")
    words = [
        f'<mark name="{word}_{i}" />{word}'
        for i, word in enumerate(words)
        if word not in ("", " ")
    ]

    ssml = " ".join(words)
    ssml = f'<speak>{ssml}<mark name="__end" /></speak>'
    return ssml


def audio_consumer(q: Queue):
    while True:
        (samples, channels) = q.get()
        play_handle = play_audio(samples, channels)
        play_handle.wait_done()


def interleave(sequential: np.ndarray):
    left = sequential[: len(sequential) // 2]
    right = sequential[len(sequential) // 2 :]
    return np.stack([left, right], axis=0).reshape(-1, order="F")


def play_audio(samples: list[float], channels: int = 1):
    if channels == 2:
        samples = interleave(samples)
    samples = samples * (2**15 - 1) / np.max(np.abs(samples))
    samples = samples.astype(np.int16)
    return sa.play_buffer(samples, channels, 2, 44100)


def main():
    load_dotenv()

    clips, resemble_project_uuid = configure_resemble()
    voice_of_mogen = init_vom()
    audio_queue = init_audio()

    while True:
        clip, word_starts, text, _ = fetch_clip(clips, resemble_project_uuid)
        audio_queue.put((clip, 1))
        print(f"clip shape: {clip.shape}")

        print("Synthesizing voice...")
        try:
            voice, _, _, _ = voice_of_mogen.process(clip, text, timestamps=word_starts)
        except Exception as e:
            print(e)
            print("Failed to synthesize voice. Skipping...")
            continue
        print("Done.")

        clip_norm = clip / np.max(np.abs(clip))
        voice_norm = voice / np.max(np.abs(voice))

        mixed = clip_norm[None, :] * 0.01 + voice_norm[:, : clip_norm.shape[-1]] * 0.9
        mixed_with_tail = np.concatenate(
            (mixed, voice_norm[:, clip_norm.shape[-1] :]), axis=1
        )

        print(f"synthesised shape: {voice.shape}")
        audio_queue.put((mixed_with_tail, 2))

        print("Done.")


if __name__ == "__main__":
    main()
