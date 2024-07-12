import logging
import math
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from itertools import chain
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import requests
import torch
from banana_dev import Client as BananaClient
from lapsolver import solve_dense

from voice_of_mogen import Sentiment
from voice_of_mogen.models import instantiate_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MAX_NOTE_LENGTH_IN_TICKS = 1000


# Note and composition representations
class Note:
    """
    A class representing a note. A note is defined by its start time, duration, pitch,
    and velocity.
    """

    def __init__(
        self,
        start_time_in_seconds: float,
        duration_in_seconds: float,
        pitch: int,
        velocity: int,
    ):
        self.start_time_in_seconds = start_time_in_seconds
        self.duration_in_seconds = duration_in_seconds
        self.pitch = pitch
        self.velocity = velocity

    def __repr__(self) -> str:
        return (
            f"Note(start_time_in_seconds={self.start_time_in_seconds}, "
            f"duration_in_seconds={self.duration_in_seconds}, pitch={self.pitch}, "
            f"velocity={self.velocity})"
        )

    def overlaps(self, other_note: "Note") -> bool:
        if self.start_time_in_seconds < other_note.start_time_in_seconds:
            return self.end_time_in_seconds > other_note.start_time_in_seconds
        else:
            return other_note.end_time_in_seconds > self.start_time_in_seconds

    @property
    def end_time_in_seconds(self) -> float:
        return self.start_time_in_seconds + self.duration_in_seconds

    def to_dict(self) -> Dict[str, float]:
        return {
            "start_time_in_seconds": self.start_time_in_seconds,
            "duration_in_seconds": self.duration_in_seconds,
            "pitch": self.pitch,
            "velocity": self.velocity,
        }


class Composition:
    """
    A class representing a composition. A composition is a list of voices, where each
    voice is a list of notes. Each note is represented by a Note object, which
    contains the note's start time, duration, pitch, and velocity.
    """

    notes: List[List[Note]]

    def __init__(self, notes: Optional[List[List[Note]]] = None):
        if notes is None:
            notes = [[]]
        else:
            self.notes = notes

    @property
    def num_voices(self) -> int:
        return len(self.notes)

    @property
    def length_in_seconds(self) -> float:
        """Returns the duration of the composition in seconds."""
        return max(
            [max([note.end_time_in_seconds for note in voice]) for voice in self.notes]
        )

    def add_note(self, voice_idx: int, note: Note):
        self._create_voices_if_necessary(voice_idx)
        self.notes[voice_idx].append(note)

    def notes_as_time_series(self, voice_idx: int, sample_rate: int) -> np.ndarray:
        ## I modified this part
        sample_rate = int(sample_rate)
        n_samples = int(self.length_in_seconds * sample_rate) + 1

        time_series = np.zeros(n_samples)
        for note in self.notes[voice_idx]:
            start_sample = int(note.start_time_in_seconds * sample_rate)
            end_sample = int(note.end_time_in_seconds * sample_rate)
            time_series[start_sample:end_sample] = note.pitch

        return time_series

    def _create_voices_if_necessary(self, voice_idx: int):
        while len(self.notes) <= voice_idx:
            self.notes.append([])

    def __repr__(self) -> str:
        return f"Composition({self.notes})"

    def offset(self, offset_in_seconds: float) -> "Composition":
        return Composition(
            notes=[
                [
                    Note(
                        start_time_in_seconds=note.start_time_in_seconds
                        + offset_in_seconds,
                        duration_in_seconds=note.duration_in_seconds,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                    for note in voice
                ]
                for voice in self.notes
            ]
        )

    def __add__(self, other: "Composition") -> "Composition":
        offset_other = other.offset(self.length_in_seconds)

        voices = []
        for voice_idx in range(max(self.num_voices, other.num_voices)):
            voice = []
            if voice_idx < self.num_voices:
                voice.extend(self.notes[voice_idx])
            if voice_idx < other.num_voices:
                voice.extend(offset_other.notes[voice_idx])
            voices.append(voice)
        return Composition(voices)

    def to_dict(self) -> Dict[str, List[List[Dict[str, float]]]]:
        return {"notes": [[note.to_dict() for note in voice] for voice in self.notes]}

    @staticmethod
    def from_dict(
        composition_dict: Dict[str, List[List[Dict[str, float]]]]
    ) -> "Composition":
        return Composition(
            notes=[
                [Note(**note_dict) for note_dict in voice]
                for voice in composition_dict["notes"]
            ]
        )

    def flatten(self) -> List[Note]:
        """Returns a flattened list of notes."""
        return [note for voice in self.notes for note in voice]

    def make_legato_(self) -> "Composition":
        sorted_notes = list(
            sorted(
                chain.from_iterable(self.notes), key=lambda x: x.start_time_in_seconds
            )
        )
        for idx, note in enumerate(sorted_notes):
            # find next non-overlapping note
            for other_note in sorted_notes[idx:]:
                if not note.overlaps(other_note):
                    note.duration_in_seconds = (
                        other_note.start_time_in_seconds - note.start_time_in_seconds
                    )

                    break

        return self

    def make_legato(self) -> "Composition":
        new_composition = deepcopy(self)
        new_composition.make_legato_()
        return new_composition

    def force_to_timestamps(self, timestamps: Sequence[float]) -> "Composition":
        new_composition = self.make_legato()

        verticals = dict()

        for voice_idx, voice in enumerate(new_composition.notes):
            for note in voice:
                end_time_in_seconds = (
                    note.start_time_in_seconds + note.duration_in_seconds
                )

                if note.start_time_in_seconds not in verticals:
                    verticals[note.start_time_in_seconds] = []
                if end_time_in_seconds not in verticals:
                    verticals[end_time_in_seconds] = []

                verticals[note.start_time_in_seconds].append(("start", note))
                verticals[end_time_in_seconds].append(("end", note))

        timestamps = deepcopy(timestamps)

        # sort vertical event timings so we can guarantee we assign a timestamp to the
        # first and last events
        keys = sorted(list(verticals.keys()))

        # shuffle the middle ones so we get some variation from the greedy assignment
        middle = np.array([keys[1:-1]])
        timestamps_middle = np.array([timestamps[1:-1]])

        costs = middle.T - timestamps_middle
        costs = np.square(costs)

        rows, cols = solve_dense(costs)
        rows = [0] + [r + 1 for r in rows] + [len(keys) - 1]
        cols = [0] + [c + 1 for c in cols] + [len(timestamps) - 1]

        for row, col in zip(rows, cols):
            time = keys[row]
            timestamp = timestamps[col]

            for boundary, note in verticals[time]:
                if boundary == "start":
                    note.start_time_in_seconds = timestamp
                elif boundary == "end":
                    note.duration_in_seconds = timestamp - note.start_time_in_seconds

        return new_composition


# Utility functions for working with notes, compositions, and token representations.
def lead_voices(notes: List[Note]) -> List[List[Note]]:
    """
    Given a list of note events, assign each note to a voice such that no
    two notes in the same voice overlap.

    Note that this is effectively a graph colouring problem, so finding an
    optimal (i.e. fewest voices) solution is NP-hard. For now, we'll just
    use a greedy algorithm and accept we may find a suboptimal solution.

    TODO: introduce pitch proximity heuristics to improve musicality of leading.

    Args:
        notes: A list of notes to assign to voices.

    Returns:
        A list of voices, each of which is a list of notes.
    """
    adjacency_matrix = np.zeros((len(notes), len(notes)))

    for i, note in enumerate(notes):
        for j, other_note in enumerate(notes[i + 1 :]):
            if note.overlaps(other_note):
                adjacency_matrix[i, j + i + 1] = 1
                adjacency_matrix[j + i + 1, i] = 1

    degrees = np.sum(adjacency_matrix, axis=0)
    voice_assignments = np.full(len(notes), -1, dtype=int)

    note_ordering = np.argsort(degrees)[::-1]

    for i in note_ordering:
        (neighbours,) = np.nonzero(adjacency_matrix[i, :])
        neighbour_assignments = voice_assignments[neighbours]

        if len(neighbour_assignments) == 0:
            voice_assignments[i] = 0
            continue

        for j in range(neighbour_assignments.max() + 2):
            if j not in neighbour_assignments:
                available_voice = j
                break

        voice_assignments[i] = available_voice

    voices = [[] for _ in range(voice_assignments.max() + 1)]

    for i, note in enumerate(notes):
        voice = voice_assignments[i]
        voices[voice].append(note)

    return voices


def tokens_to_events(
    tokens: torch.Tensor, token_mappings: dict
) -> List[Tuple[str, int]]:
    """
    Converts a sequence of tokens to a sequence of events.

    Args:
        tokens: A tensor of shape (seq_len, 1) containing the tokens to convert.

    Returns:
        A list of tuples of the form (event_type, event_value).
    """
    events = []

    for i in range(tokens.shape[-1]):
        event = token_to_event(tokens[..., i], token_mappings)
        events.append(event)

    return events


def token_to_event(token: torch.Tensor, token_mappings: dict) -> Tuple[str, int]:
    """
    Converts a token to an event.

    Args:
        token: A tensor of shape (1, 1) containing the token to convert.

    Returns:
        A tuple of the form (event_type, event_value).
    """
    token_idx = token.item()

    event = token_mappings["idx2tuple"][token_idx]
    if event[0] == "<":
        return (event, None)

    event_type_idx, event_value = event
    event_type_str = token_mappings["idx2event"][event_type_idx]

    return event_type_str, event_value


def events_to_notes(
    events: List[Tuple[str, int]],
    duration_in_seconds: Optional[float] = None,
    ticks_per_second: Optional[float] = None,
    return_type: Literal["notes", "active_notes"] = "notes",
) -> List[Note]:
    """
    Given a list of events, returns a list of notes.

    Args:
        events: A list of events, where each event is a tuple of the form
            (event_type, event_value).

    Returns:
        A list of notes.
    """
    if return_type == "notes" and (
        duration_in_seconds is None or ticks_per_second is None
    ):
        raise ValueError(
            "If return_type is 'notes', duration_in_seconds and ticks_per_second must "
            "be specified."
        )

    active_notes = dict()
    notes = []
    current_time = 0.0

    for event_str, event_value in events:
        if event_str[:2] == "ON" and "DRUM" not in event_str:
            if event_value in active_notes:
                continue
            else:
                active_notes[event_value] = current_time

        if event_str[:3] == "OFF":
            if event_value in active_notes:
                if return_type == "notes":
                    start_time = active_notes[event_value]
                    duration = current_time - start_time

                    note = Note(
                        start_time / ticks_per_second,
                        duration / ticks_per_second,
                        event_value,
                        100,
                    )
                    notes.append(note)

                del active_notes[event_value]
            else:
                continue

        if event_str == "TIMESHIFT":
            current_time += event_value

    # If we're using the active notes return type, we exit early, simply returning
    # a list of currently held notes.
    if return_type == "active":
        return active_notes

    # clean up any remaining active notes
    for note_id, start_time in active_notes.items():
        duration = duration_in_seconds - start_time / ticks_per_second
        note = Note(start_time / ticks_per_second, duration, note_id, 100)
        notes.append(note)

    return notes


def _make_time_shifts(time_to_fill_in_ticks: float) -> List[Tuple[str, int]]:
    time_shifts = []

    if time_to_fill_in_ticks > 0:
        time_shifts += [("TIMESHIFT", MAX_NOTE_LENGTH_IN_TICKS)] * (
            int(time_to_fill_in_ticks // MAX_NOTE_LENGTH_IN_TICKS)
        )
        remainder = (
            math.ceil((time_to_fill_in_ticks % MAX_NOTE_LENGTH_IN_TICKS) / 8) * 8
        )
        time_shifts += [("TIMESHIFT", remainder)]

    return time_shifts


def notes_to_events(
    notes: List[Note], ticks_per_second: float, inst_str: str = "PIANO"
) -> List[Tuple[str, int]]:
    """
    Given a list of notes, return a list of events.

    Args:
        notes: A list of notes.

    Returns:
        A list of events, where each event is a tuple of the form
        (event_type, event_value).
    """
    active_notes = dict()
    events = []
    current_time_in_ticks = 0.0

    on_str = f"ON_{inst_str}"
    off_str = f"OFF_{inst_str}"

    notes = sorted(notes, key=lambda x: x.start_time_in_seconds)

    for note in notes:
        start_time = int(note.start_time_in_seconds * ticks_per_second)
        duration = int(note.duration_in_seconds * ticks_per_second)
        end_time = start_time + duration

        if start_time == current_time_in_ticks:
            # If we're already at the note's onset time we...

            # 1. Add a note on event
            events.append((on_str, note.pitch))
            # 2. Update our active notes map to make sure we add a note off
            active_notes[note.pitch] = end_time
        elif start_time > current_time_in_ticks:
            # If we're not at the note's onset time we...

            # 1. Make a list of note-offs required between now and the onset time
            notes_to_close = [
                (pitch, end_time)
                for pitch, end_time in active_notes.items()
                if current_time_in_ticks < end_time <= start_time
            ]
            notes_to_close = sorted(notes_to_close, key=lambda x: x[1])

            # 2. Add these note-offs, incrementing our counter as we go
            for pitch, end_time in notes_to_close:
                time_shift_amount = end_time - current_time_in_ticks
                time_shifts = _make_time_shifts(time_shift_amount)
                events.extend(time_shifts)
                current_time_in_ticks += time_shift_amount
                events.append((off_str, pitch))

                del active_notes[pitch]

            # 3. Add a time shift event if necessary
            time_shift_amount = start_time - current_time_in_ticks
            time_shifts = _make_time_shifts(time_shift_amount)
            events.extend(time_shifts)
            current_time_in_ticks += time_shift_amount

            # 4. Add a note on event
            events.append((on_str, note.pitch))
            active_notes[note.pitch] = end_time
        else:
            raise ValueError(
                f"Note start time {start_time} is less than current time "
                f"{current_time_in_ticks}. This should never happen. "
            )

    # Finally, we clean up any remaining active notes
    remaining_notes = sorted(active_notes.items(), key=lambda x: x[1])
    for pitch, end_time in remaining_notes:
        time_shift_amount = end_time - current_time_in_ticks
        time_shifts = _make_time_shifts(time_shift_amount)
        events.extend(time_shifts)
        current_time_in_ticks += time_shift_amount
        events.append((off_str, pitch))

    return events


def events_to_tokens(
    events: List[Tuple[str, int]], token_mappings: dict
) -> torch.Tensor:
    """
    Given a list of events, return a tensor representation.

    Args:
        events: A list of events, where each event is a tuple of the form
            (event_type, event_value).
        token_mappings: A dictionary mapping event types to token IDs.

    Returns:
        A tensor representation of the events.
    """
    events = [
        (token_mappings["event2idx"][event_str], event_value)
        for event_str, event_value in events
    ]
    tokens = [
        token_mappings["tuple2idx"][event]
        for event in events
        # silently ignore invalid tokens
        if event in token_mappings["tuple2idx"]
    ]
    return torch.tensor(tokens, dtype=torch.long)[None]


def notes_to_composition(notes: List[Note]) -> Composition:
    """
    Given a list of note events, return a Composition object containing
    the notes separated into voices.

    Args:
        notes: A list of notes to assign to voices.

    Returns:
        A Composition object containing the notes separated into voices.
    """
    voices = lead_voices(notes)

    return Composition(voices)


def events_to_composition(
    events: List[Tuple[str, int]],
    duration_in_seconds: float,
    ticks_per_second: float,
) -> Composition:
    """
    Given a list of events, return a Composition object containing
    the notes separated into voices.

    Args:
        events: A list of events, where each event is a tuple of the form
            (event_type, event_value).

    Returns:
        A Composition object containing the notes separated into voices.
    """
    notes = events_to_notes(events, duration_in_seconds, ticks_per_second)
    return notes_to_composition(notes)


def composition_to_events(
    composition: Composition,
    ticks_per_second: float,
) -> List[Tuple[str, int]]:
    """
    Given a Composition object, return a list of events.

    Args:
        composition: A Composition object.
        ticks_per_second: The number of ticks per second.

    Returns:
        A list of events, where each event is a tuple of the form
        (event_type, event_value).
    """
    notes = composition.flatten()
    return notes_to_events(notes, ticks_per_second)


class ISamplingRule(ABC):
    """
    An interface for classes that constrain multinomial sampling from a softmax
    distribution.

    Rather than maintain internal state, sampling rules should make use of the context
    tensor which is passed into the apply() method.
    """

    @abstractmethod
    def apply(
        self,
        context: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def update(self, event: Tuple[str, int]):
        pass

    def reset(self):
        pass


class Temperature(ISamplingRule):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def apply(
        self,
        context: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return logits / self.temperature


class PolyphonyRange(ISamplingRule):
    """
    A sampling rule that enforces a minimum and maximum number of voices in the
    generated composition
    """

    def __init__(self, min_num_voices: int, max_num_voices: int, token_mappings: dict):
        self.token_mappings = token_mappings
        self.min_num_voices = min_num_voices
        self.max_num_voices = max_num_voices

        # TODO: refactor into a separate (more readable) method
        self.timeshift_tokens = [
            idx
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] == 10
        ]
        self.off_tokens = [
            idx
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<" and symbol[0] % 2 == 0 and symbol[0] != 10
        ]
        self.on_tokens = [
            idx
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<" and symbol[0] % 2 == 1
        ]

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        context_events = tokens_to_events(context, self.token_mappings)
        active_notes = events_to_notes(context_events, return_type="active")
        num_active_notes = len(active_notes.keys())

        if num_active_notes < self.min_num_voices:
            # if there are not enough active notes, we want to prevent the model
            # from generating any off tokens or timeshift tokens
            for token_idx in self.timeshift_tokens + self.off_tokens:
                logits[:, token_idx] = float("-inf")

        if num_active_notes == self.max_num_voices:
            # there are enough voices already. we want to prevent any ON tokens
            # from being generated
            for token_idx in self.on_tokens:
                logits[:, token_idx] = float("-inf")

        if num_active_notes > self.max_num_voices:
            # there are too many voices. we want to only allow note offs.
            for token_idx in self.on_tokens + self.timeshift_tokens:
                logits[:, token_idx] = float("-inf")

        return logits


class PreventInvalidNoteOffs(ISamplingRule):
    """
    A sampling rule that prevents the model from generating note off events for
    notes that are not currently active.
    """

    def __init__(self, token_mappings: dict):
        self.token_mappings = token_mappings

        self.off_tokens = [
            (idx, symbol[1])
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<" and symbol[0] % 2 == 0 and symbol[0] != 10
        ]

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        context_events = tokens_to_events(context, self.token_mappings)
        active_notes = events_to_notes(context_events, return_type="active")

        for token_idx, note_id in self.off_tokens:
            if note_id not in active_notes:
                logits[:, token_idx] = float("-inf")

        return logits


class PreventInvalidNoteOns(ISamplingRule):
    """
    A sampling rule that prevents the model from generating note on events for
    notes that are already active
    """

    def __init__(self, token_mappings: dict):
        self.token_mappings = token_mappings

        self.on_tokens = [
            (idx, symbol[1])
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<" and symbol[0] % 2 == 1
        ]

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        context_events = tokens_to_events(context, self.token_mappings)
        active_notes = events_to_notes(context_events, return_type="active")

        for token_idx, note_id in self.on_tokens:
            if note_id in active_notes:
                logits[:, token_idx] = float("-inf")

        return logits


class NoteRange(ISamplingRule):
    """
    A sampling rule that enforces a minimum and maximum note range in the generated
    composition.
    """

    def __init__(self, min_note: int, max_note: int, token_mappings: dict):
        self.token_mappings = token_mappings
        self.min_note = min_note
        self.max_note = max_note

        self.excluded_tokens = [
            idx
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<"
            and (symbol[1] < self.min_note or symbol[1] > self.max_note)
        ]

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for token_idx in self.excluded_tokens:
            logits[:, token_idx] = float("-inf")

        return logits


class NucleusSampling(ISamplingRule):
    """
    Nucleus sampling (also known as top-p sampling) is a technique for sampling from
    a softmax distribution that only considers the top p probability mass of the
    distribution. This allows us to avoid sampling from unlikely events while still
    allowing for some randomness in the generated composition.
    """

    def __init__(self, p: float):
        self.p = p

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        softmax = torch.softmax(logits, dim=-1)
        sorted_softmax, sorted_indices = torch.sort(softmax, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_softmax, dim=-1)
        index_mask = cumulative_probs > self.p

        # shift the mask to the right by one so that the first index is always included
        index_mask[:, 1:] = index_mask[:, :-1].clone()
        index_mask[:, 0] = False

        log.info(f"Omitting {index_mask.int().sum()} tokens through NucleusSampling.")

        # zero out all indices that are not in the top p
        indices_to_remove = sorted_indices[index_mask]
        logits[:, indices_to_remove] = float("-inf")

        return logits


class TopKSampling(ISamplingRule):
    def __init__(self, k: float):
        self.k = k

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        _, indices = logits.topk(self.k, dim=-1)
        mask = torch.full_like(logits, True, dtype=torch.bool)
        mask[..., indices] = False
        logits[mask] = float("-inf")
        return logits


class MinNoteLength(ISamplingRule):
    """
    A sampling rule that enforces a minimum note length in the generated composition.
    """

    def __init__(
        self,
        min_length_in_seconds: float,
        ticks_per_second: float,
        token_mappings: dict,
    ):
        self.min_length_in_ticks = min_length_in_seconds * ticks_per_second
        self.token_mappings = token_mappings

        self.off_tokens = [
            (idx, symbol[1])
            for symbol, idx in self.token_mappings["tuple2idx"].items()
            if symbol[0] != "<" and symbol[0] % 2 == 0 and symbol[0] != 10
        ]

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        context_events = tokens_to_events(context, self.token_mappings)
        active_notes = events_to_notes(context_events, return_type="active")

        note_durations = dict()

        for pitch in active_notes.keys():
            time_since_note_on = 0.0
            for event_str, event_value in context_events[::-1]:
                if event_str[:2] == "ON" and event_value == pitch:
                    break
                elif event_str == "TIMESHIFT":
                    time_since_note_on += event_value
            note_durations[pitch] = time_since_note_on

        for token_idx, pitch in self.off_tokens:
            if (
                pitch in active_notes
                and note_durations[pitch] < self.min_length_in_ticks
            ):
                logits[:, token_idx] = float("-inf")

        return logits


class ExclusionSamplingRule(ISamplingRule):
    """
    Base class for exclusion-based sampling rules -- i.e. those that set negative
    infinity log prob for some fixed list of tokens.
    """

    def __init__(self, token_mappings: dict):
        self.token_mappings = token_mappings

        self.symbols_to_exclude = self.make_exclusion_list()

    def make_exclusion_list(self) -> Sequence[int]:
        raise NotImplementedError

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for symbol_to_exclude in self.symbols_to_exclude:
            idx_to_exclude = self.token_mappings["tuple2idx"][symbol_to_exclude]

            if idx_to_exclude < logits.shape[-1]:
                logits[:, idx_to_exclude] = float("-inf")

        return logits


class StripSpecialTokens(ExclusionSamplingRule):
    """
    A sampling rule that excludes special tokens, i.e. those that begin with "<"
    """

    def make_exclusion_list(self) -> Sequence[int]:
        return [
            symbol
            for symbol in self.token_mappings["tuple2idx"].keys()
            if symbol[0] == "<"
        ]


class ExcludeTokens(ExclusionSamplingRule):
    """
    A sampling rule that excludes certain event IDs. This allows us to, e.g. limit
    generations to a single instrument.
    """

    def __init__(self, events_to_exclude: Sequence[int], token_mappings: dict):
        self.events_to_exclude = events_to_exclude
        super().__init__(token_mappings)

    def make_exclusion_list(self) -> Sequence[int]:
        return [
            symbol
            for symbol in self.token_mappings["tuple2idx"].keys()
            if symbol[0] in self.events_to_exclude
        ]


class PenalizeRepetition(ISamplingRule):
    """
    A sampling rule that penalizes token repetitions. This is done by maintaining
    a dictionary of decaying penalties for each token. Each time a token is
    generated, its penalty is incremented by a fixed amount. The penalty decays
    by a fixed factor each time step.
    """

    def __init__(self, token_mappings: dict, penalty: float = 1.0, decay: float = 0.99):
        self.token_mappings = token_mappings
        self.penalties = dict()
        self.penalty = penalty
        self.decay = decay

    def apply(self, context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for token, penalty in self.penalties.items():
            candidate_events = [
                k for k in self.token_mappings["tuple2idx"].keys() if token[0] in k
            ]

            # if we are not processing a time shift, we want to penalize all
            # instruments for playing the same note
            for candidate_event in candidate_events:
                event_idx = self.token_mappings["event2idx"][candidate_event]
                idx = self.token_mappings["tuple2idx"][(event_idx, token[1])]

                logit_val = logits[:, idx]

                if not logit_val.isneginf():
                    logits[:, idx] = logits[:, idx] / (1 + penalty)

        return logits

    def _event_str_to_key(self, event: Tuple[str, int]) -> Tuple[str, int]:
        if event[0] == "TIMESHIFT":
            return event
        elif event[0][:2] == "ON":
            return ("ON", event[1] % 12)
        else:
            return None

    def update(self, event: Tuple[str, int]):
        for token, penalty in self.penalties.items():
            self.penalties[token] = penalty * self.decay

        key = self._event_str_to_key(event[0])
        if key in self.penalties:
            self.penalties[key] += self.penalty
        elif key is not None:
            self.penalties[key] = self.penalty

    def reset(self):
        self.penalties = dict()


class IComposer(ABC):
    """
    An interface for classes that generate sentiment-conditioned compositions.
    """

    @abstractmethod
    def compose(
        self,
        sentiment: Sentiment,
        duration_in_seconds: float,
        timestamps: Optional[List[float]],
        context: Optional[Composition],
        composition_parameters: Optional[Dict[str, Any]],
    ) -> Composition:
        pass


class APIComposer(IComposer):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def compose(
        self,
        sentiment: Sentiment,
        duration_in_seconds: float,
        timestamps: Optional[List[float]],
        context: Optional[Composition],
        composition_parameters: Optional[Dict[str, Any]] = None,
    ) -> Composition:
        sentiment = sentiment.to_dict()
        if context is not None:
            context = context.to_dict()

        request = {
            "sentiment": sentiment,
            "duration_in_seconds": duration_in_seconds,
            "timestamps": timestamps,
            "context": context,
            "composition_parameters": composition_parameters or {},
        }
        log.info("This is API composing...", self.endpoint_url)
        print(request)
        response = requests.post(self.endpoint_url, json=request)
        response.raise_for_status()

        composition_dict = response.json()["composition"]
        composition = Composition.from_dict(composition_dict)
        return composition


class BananaComposer(IComposer):
    def __init__(self, endpoint_url: str, api_key: str, model_key: str):
        self.client = BananaClient(
            model_key=model_key,
            api_key=api_key,
            url=endpoint_url,
        )

    def compose(
        self,
        sentiment: Sentiment,
        duration_in_seconds: float,
        timestamps: Optional[List[float]],
        context: Optional[Composition],
        composition_parameters: Optional[Dict[str, Any]] = None,
    ) -> Composition:
        sentiment = sentiment.to_dict()
        if context is not None:
            context = context.to_dict()

        request = {
            "sentiment": sentiment,
            "duration_in_seconds": duration_in_seconds,
            "timestamps": timestamps,
            "context": context,
            "composition_parameters": composition_parameters or {},
        }

        response, _ = self.client.call("/compose", request)

        composition_dict = response["composition"]
        composition = Composition.from_dict(composition_dict)
        return composition


class MusicTransformerComposer(IComposer):
    """
    A composer that generates compositions using the MusicTransformer model.

    Args:
        checkpoint_path: The path to the MusicTransformer model's checkpoint directory.
        temperature: The temperature to use when sampling from the model's output
            distribution. Higher temperatures result in more random samples.
        max_input_length: The maximum number of tokens to feed into the model. If the
            input text is longer than this, it will be truncated from the front.
        use_automatic_mixed_precision: Whether to use automatic mixed precision, which
            can speed up training and inference on GPUs with Tensor Cores.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/midi-emotion-finetuned",
        temperature: float = 0.8,
        max_input_length: int = 512,
        use_automatic_mixed_precision: bool = True,
        ticks_per_second: float = 1250.0,
        device: str = "cpu",
    ):
        self.device = device
        log.info(f"({self.__class__.__name__}) Loading model...")
        self.model, self.token_mappings = instantiate_model(checkpoint_path)
        self.model.eval()
        self.model.to(device)
        log.info(f"({self.__class__.__name__}) Model loaded.")

        log.info(f"({self.__class__.__name__}) Setting up sampling rules...")
        # TODO: setup via dependency injection
        self.sampling_rules = [
            StripSpecialTokens(self.token_mappings),
            # ExcludeTokens((0, 1), self.token_mappings),
            # PolyphonyRange(1, 1e8, self.token_mappings),
            # NoteRange(36, 80, self.token_mappings),
            # PreventInvalidNoteOffs(self.token_mappings),
            # PreventInvalidNoteOns(self.token_mappings),
            # MinNoteLength(0.01, ticks_per_second, self.token_mappings),
            Temperature(temperature),
            PenalizeRepetition(self.token_mappings, penalty=0.2, decay=1.0),
            # NucleusSampling(0.99),
            # TopKSampling(10),
        ]

        self.max_input_length = max_input_length
        self.use_automatic_mixed_precision = use_automatic_mixed_precision
        self.ticks_per_second = ticks_per_second

    @torch.no_grad()
    def compose(
        self,
        sentiment: Sentiment,
        duration_in_seconds: float,
        timestamps: Optional[Sequence[float]] = None,
        max_tokens: int = 1024,
        context: Optional[Composition] = None,
        composition_parameters: Optional[Dict[str, Any]] = None,
    ) -> Composition:
        """
        Produce a composition from the MusicTransformer model. The composition is
        generated by autoregressively sampling from the model's output distribution
        at each timestep.

        Args:
            sentiment: The sentiment to condition the composition on.
            duration_in_seconds: The duration of the composition in seconds.
            timestamps: Timestamps of word or phoneme boundaries in the given text. If
                provided, the sampling will be biased to align with these timestamps.
            max_tokens: The maximum number of tokens to generate.
        """
        if composition_parameters is not None:
            scale_duration = composition_parameters.get("scale_duration", 1.0)
        else:
            scale_duration = 1.0
        log.info(
            f"({self.__class__.__name__}) Composing {duration_in_seconds} "
            f"seconds of music..."
        )
        log.info(
            f"({self.__class__.__name__}) Using scale factor {scale_duration:.2f} "
            f"giving effective duration of {duration_in_seconds * scale_duration:.2f} "
            "seconds."
        )

        max_ticks = duration_in_seconds * self.ticks_per_second * scale_duration

        for sampling_rule in self.sampling_rules:
            sampling_rule.reset()

        events = []
        generated_song = torch.tensor([[]], device=self.device, dtype=torch.long)
        conditioning_input = torch.tensor(
            [[sentiment.quantized_valence, sentiment.quantized_arousal]],
            device=self.device,
        )

        if context is None:
            primers = [["<START>"]]
            primer_indices = [
                [self.token_mappings["tuple2idx"][token] for token in primer]
                for primer in primers
            ]
            generated_indices = torch.tensor(
                primer_indices, device=self.device, dtype=torch.long
            )
        else:
            context_events = composition_to_events(context, self.ticks_per_second)
            generated_indices = events_to_tokens(
                context_events, self.token_mappings
            ).to(self.device)

        generated_time = 0.0

        for i in range(max_tokens):
            log.debug(
                f"Generating next token. {generated_time} ticks generated so far."
            )

            generated_song = torch.cat([generated_song, generated_indices], dim=-1)
            context = generated_song
            if generated_song.shape[-1] > self.max_input_length:
                context = generated_song[:, -self.max_input_length :]

            generated_indices = self._sample_next_token(context, conditioning_input)
            event_str, event_value = token_to_event(
                generated_indices.squeeze(), self.token_mappings
            )

            for sampling_rule in self.sampling_rules:
                sampling_rule.update((event_str, event_value))

            if event_str == "TIMESHIFT":
                generated_time += event_value

            # workaround for initially sampling silence in continued compositions.
            # this will be superceded by cost-based time synchronization.
            if not (len(events) == 0 and event_str[:2] in ("TI", "OF")):
                events.append((event_str, event_value))

            if generated_time >= max_ticks:
                break

        composition = events_to_composition(
            events, duration_in_seconds, self.ticks_per_second * scale_duration
        )
        if timestamps is not None:
            composition = composition.force_to_timestamps(timestamps)

        return composition

    def _sample_next_token(
        self,
        context: torch.Tensor,
        conditioning_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample the next token from the model's output distribution.
        """
        logits = self.model(context, conditioning_input)
        logits = logits[:, -1, :]
        logits = torch.nan_to_num(logits, 0.0)

        if torch.all(logits == 0.0):
            logits = torch.ones_like(logits)

        for sampling_rule in self.sampling_rules:
            logits = sampling_rule.apply(context, logits)

        # softmax with temperature
        distribution = torch.softmax(logits, dim=-1)

        sample = torch.multinomial(distribution, num_samples=1)

        return sample
