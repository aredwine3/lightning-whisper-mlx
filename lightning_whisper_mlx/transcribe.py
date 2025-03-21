# Copyright © 2023 Apple Inc.

import sys
import warnings
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import tqdm
import time

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .load_models import load_model
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer


def _format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


class ModelHolder:
    model = None
    model_path = None

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        if cls.model is None or model_path != cls.model_path:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
        return cls.model


def transcribe_audio(
    audio: Union[str, np.ndarray, mx.array],
    *,
    path_or_hf_repo: str = "mlx-community/whisper-tiny",
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1, 
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    batch_size: 6,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array]
        The path to the audio file to open, or the audio waveform

    path_or_hf_repo: str
        The localpath to the Whisper model or HF Hub repo with the MLX converted weights.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
    model = ModelHolder.get_model(path_or_hf_repo, dtype)

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES

    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":
            make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
                system_encoding
            )
        else:
            make_safe = lambda x: x

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` decoding option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")
    
    def decode_process(segment_batch, t):
        kwargs = {**decode_options}
        options = DecodingOptions(**kwargs, temperature=t)
        decode_results = model.decode(segment_batch, options)
        return decode_results

    def decode_with_fallback(segment_batch: mx.array) -> DecodingResult:
        decode_results = decode_process(segment_batch, 0.0)
        final_decode = []

        for i, decode_result in enumerate(decode_results):
            segment = segment_batch[i:i+1, :, :]  
            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True

            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  

            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  

            if needs_fallback:
                final_decode.append(decode_process(segment, 1.0)[0])
            else:
                final_decode.append(decode_result)

        return final_decode

    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    input_stride = N_FRAMES // model.dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *, start: float, end: float, tokens: mx.array, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": res.temperature,
            "avg_logprob": res.avg_logprob,
            "compression_ratio": res.compression_ratio,
            "no_speech_prob": res.no_speech_prob,
        }
    
    def format_output(tokens, res):
        seek = 0
        current_segments = []

        if no_speech_threshold is not None:
            should_skip = res.no_speech_prob > no_speech_threshold
            if (
                logprob_threshold is not None
                and res.avg_logprob > logprob_threshold
            ):
                should_skip = False

            if should_skip:
                seek += (
                    segment_size 
                )
                return current_segments, seek

        def word_anomaly_score(word: dict) -> float:
            probability = word.get("probability", 0.0)
            duration = word["end"] - word["start"]
            score = 0.0
            if probability < 0.15:
                score += 1.0
            if duration < 0.133:
                score += (0.133 - duration) * 15
            if duration > 2.0:
                score += duration - 2.0
            return score

        def is_segment_anomaly(segment: Optional[dict]) -> bool:
            if segment is None or not segment["words"]:
                return False
            words = [
                w for w in segment["words"] if w["word"] not in punctuation
            ]
            words = words[:8]
            score = sum(word_anomaly_score(w) for w in words)
            return score >= 3 or score + 0.01 >= len(words)

        def next_words_segment(segments: List[dict]) -> Optional[dict]:
            return next((s for s in segments if s["words"]), None)

        timestamp_tokens = tokens >= tokenizer.timestamp_begin
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [
            False,
            True,
        ]

        consecutive = np.where(
            np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
        )[0]
        
        if len(consecutive) > 0:
            slices = consecutive.tolist()
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_pos = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_pos = (
                    sliced_tokens[-1].item() - tokenizer.timestamp_begin
                )
                current_segments.append(
                    new_segment(
                        start=time_offset
                        + start_timestamp_pos * time_precision,
                        end=time_offset + end_timestamp_pos * time_precision,
                        tokens=sliced_tokens,
                        result=res,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                seek += segment_size
            else:
                last_timestamp_pos = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_pos * input_stride
        else:
            duration = segment_duration
            #timestamps = tokens[timestamp_tokens.nonzero()[0]] # AttributeError: 'mlx.core.array' object has no attribute 'nonzero'
            #timestamps = tokens[np.asarray(timestamp_tokens).nonzero()[0]]
            np_tokens = np.asarray(tokens)
            timestamps = mx.array(np_tokens[np.asarray(timestamp_tokens).nonzero()[0]])
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != tokenizer.timestamp_begin
            ):
                last_timestamp_pos = (
                    timestamps[-1].item() - tokenizer.timestamp_begin
                )
                duration = last_timestamp_pos * time_precision

            current_segments.append(
                new_segment(
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                    result=res,
                )
            )
            seek += segment_size

        for i, segment in enumerate(current_segments):
            if (
                segment["start"] == segment["end"]
                or segment["text"].strip() == ""
            ):
                segment["text"] = ""
                segment["tokens"] = []
                segment["words"] = []
        
        return current_segments, seek

    seek_clip_end = seek_clips[0][1]
    seek = -3000
    while seek < seek_clip_end:
        time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)

        mel_segments = []
        mel_timestamps = []

        for _ in range(batch_size):
            seek +=  N_FRAMES
            if seek > seek_clip_end:
                break
            segment_size = min(
                N_FRAMES, content_frames - seek, seek_clip_end - seek
            )
            mel_segment = mel[seek : seek + segment_size]

            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)
            mel_segments.append(mel_segment)
            mel_timestamps.append((seek, seek + segment_size))
        
        if not len(mel_segments):
            break

        mel_segment_batch = mx.array(mx.stack(mel_segments, axis=0))
        decode_options["prompt"] = all_tokens[prompt_reset_since:]
        result: DecodingResult = decode_with_fallback(mel_segment_batch)

        for index, res in enumerate(result):
            start_seek, end_seek = mel_timestamps[index]

            tokens = mx.array(res.tokens)
            current_segments, value_seek = format_output(tokens, res) 

            tokens =  [token
                    for segment in current_segments
                    for token in segment["tokens"]
            ]

            all_segments.append([start_seek, end_seek,tokenizer.decode(tokens)])
               
            all_tokens.extend(
                [
                    token
                    for segment in current_segments
                    for token in segment["tokens"]
                ]
            )

            if not condition_on_previous_text or res.temperature > 0.5:
                prompt_reset_since = len(all_tokens)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
