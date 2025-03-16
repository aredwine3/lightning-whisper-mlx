# Copyright © 2023 Apple Inc.

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx
#import numba
import numpy as np
from scipy import signal

from .audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .model import Whisper


def median_filter_OG(x: np.ndarray, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        # F.pad requires the padding width to be smaller than the input dimension
        return x

    if (ndim := x.ndim) <= 2:
        # `F.pad` does not support 1D or 2D inputs for reflect padding but supports 3D and 4D
        x = x[None, None, :]

    assert (
        filter_width > 0 and filter_width % 2 == 1
    ), "`filter_width` should be an odd number"

    x = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")

    # todo: more efficient version in mlx
    result = signal.medfilt(x.astype(np.float32), kernel_size=(1, 1, filter_width))[
        ..., pad_width:-pad_width
    ]

    if ndim <= 2:
        result = result[0, 0]

    return result

@mx.compile
def median_filter(x: mx.array, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x` using MLX"""
    pad_width = filter_width // 2
    
    if x.shape[-1] <= pad_width:
        return x
    
    original_shape = x.shape
    original_ndim = len(original_shape)
    
    # Handle lower dimensional inputs
    if original_ndim <= 2:
        x = x.reshape(1, 1, *original_shape) if original_ndim > 0 else x.reshape(1, 1, 1)
    
    # Pad the array
    x_padded = mx.pad(x, [(0, 0), (0, 0), (pad_width, pad_width)], mode="reflect")
    
    # Apply sliding window to get all filter windows
    result = []
    for i in range(filter_width, x_padded.shape[-1] + 1):
        window = x_padded[..., i-filter_width:i]
        # Sort values in each window and take the middle (median)
        sorted_window = mx.sort(window, axis=-1)
        median = sorted_window[..., pad_width]
        result.append(median)
    
    result = mx.stack(result, axis=-1)
    
    # Restore original shape
    if original_ndim <= 2:
        if original_ndim == 0:
            result = result[0, 0, 0]
        elif original_ndim == 1:
            result = result[0, 0]
        else:  # original_ndim == 2
            result = result[0]
            
    return result

"""
@numba.jit(nopython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


@numba.jit(nopython=True, parallel=True)
def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)
"""

def dtw(x: np.ndarray) -> np.ndarray:
    # todo: more efficient version in mlx
    #return dtw_cpu(x)
    return dtw_mlx(x)



@mx.compile  # compile for performance (MLX’s version of JIT)
def backtrace_mlx_OG(trace: mx.array):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    # Note: using a Python list in a compiled function might not be ideal.
    # In a production version you might preallocate a result array.
    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")
    res = mx.array(result)
    return res[::-1, :].T



@mx.compile
def dtw_mlx_OG(x: mx.array):
    N, M = x.shape
    # Create cost and trace arrays with MLX functions.
    cost = mx.ones((N + 1, M + 1), dtype=mx.float32) * mx.inf
    trace = -mx.ones((N + 1, M + 1), dtype=mx.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace_mlx(trace)

@mx.compile
def dtw_mlx(x: mx.array, use_vmap: bool = True, use_wavefront: bool = False):
    N, M = x.shape
    cost = mx.full((N + 1, M + 1), mx.inf, dtype=mx.float32)
    cost = cost.at[0, 0].set(0)
    trace = -mx.ones((N + 1, M + 1), dtype=mx.float32)
    
    if use_vmap:
        # Create a vectorized version of process_cell
        vmap_process = mx.vmap(process_cell, in_axes=(0, 0))
        
        for i in range(1, N + 1):
            # For each position j in row i, gather the three previous costs
            diag_costs = cost[i-1, :M]
            above_costs = cost[i-1, 1:M+1] 
            left_costs = cost[i, :M]
            
            # Stack the costs for each position
            prev_costs = mx.stack([diag_costs, above_costs, left_costs], axis=1)
            x_vals = x[i-1, :M]
            
            # Process all cells in the row with vmap
            indices, new_costs = vmap_process(prev_costs, x_vals)
        
        # Update matrices
        for j in range(1, M + 1):
            idx = j - 1
            cost = cost.at[i, j].set(new_costs[idx])
            trace = trace.at[i, j].set(indices[idx])
            
    elif use_wavefront:
        # Process the matrix in anti-diagonal wavefronts
        for wave in range(1, N + M):
            # Calculate valid i,j positions on this wavefront
            i_values = mx.arange(max(1, wave - M + 1), min(wave, N) + 1)
            j_values = wave + 1 - i_values
            
            # Filter valid indices
            valid = (j_values >= 1) & (j_values <= M)
            if not mx.any(valid):
                continue
                
            i_indices = i_values[valid]
            j_indices = j_values[valid]
            
            # Extract costs for each position in the wavefront
            diag_indices = mx.stack([i_indices - 1, j_indices - 1], axis=1)
            above_indices = mx.stack([i_indices - 1, j_indices], axis=1)
            left_indices = mx.stack([i_indices, j_indices - 1], axis=1)
            
            # Get the three directions' costs using vectorized operations
            diag_costs = mx.array([cost[i.item(), j.item()] for i, j in zip(diag_indices[:, 0], diag_indices[:, 1])])
            above_costs = mx.array([cost[i.item(), j.item()] for i, j in zip(above_indices[:, 0], above_indices[:, 1])])
            left_costs = mx.array([cost[i.item(), j.item()] for i, j in zip(left_indices[:, 0], left_indices[:, 1])])
            
            # Get corresponding x values
            x_values = mx.array([x[i.item()-1, j.item()-1] for i, j in zip(i_indices, j_indices)])
            
            # Find min costs in one vectorized operation
            all_costs = mx.stack([diag_costs, above_costs, left_costs], axis=-1)
            min_indices = mx.argmin(all_costs, axis=-1)
            min_costs = mx.take_along_axis(all_costs, min_indices[:, None], axis=-1).squeeze(-1)
            
            # Update matrices for all cells in this wavefront
            for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                cost = cost.at[i.item(), j.item()].set(x_values[idx] + min_costs[idx])
                trace = trace.at[i.item(), j.item()].set(min_indices[idx])
    
    return backtrace_mlx(trace)

@mx.compile
def process_cell(prev_costs, x_val):
    # prev_costs contains [diagonal, above, left]
    min_idx = mx.argmin(prev_costs)
    return min_idx, x_val + prev_costs[min_idx]

@mx.compile
def backtrace_mlx(trace: mx.array):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace = trace.copy()
    trace = trace.at[0, :].set(2)
    trace = trace.at[:, 0].set(1)

    # Pre-allocate maximum size
    max_steps = i + j
    result_i = mx.zeros(max_steps, dtype=mx.int32)
    result_j = mx.zeros(max_steps, dtype=mx.int32)
    
    # Fill arrays in one step where possible
    step = 0
    while i > 0 or j > 0:
        result_i = result_i.at[step].set(i-1)
        result_j = result_j.at[step].set(j-1)
        step += 1
        
        t = trace[i, j]
        # Use vectorized conditional movement
        di = mx.array([1, 1, 0])[t.astype(mx.int32)]
        dj = mx.array([1, 0, 1])[t.astype(mx.int32)]
        i -= di
        j -= dj
    
    # Create final result with efficient slicing and reversal
    result_i = result_i[:step][::-1]
    result_j = result_j[:step][::-1]
    
    return mx.stack([result_i, result_j])


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: mx.array,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    if len(text_tokens) == 0:
        return []

    tokens = mx.array(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    )

    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens[None, :])
    # consider only the logits associated with predicting text
    sampled_logits = logits[0][len(tokenizer.sot_sequence) : -2, : tokenizer.eot]
    token_probs = mx.softmax(sampled_logits.astype(mx.float32), axis=-1).astype(
        sampled_logits.dtype
    )
    text_token_probs = mx.take_along_axis(
        token_probs, mx.array(text_tokens)[:, None], axis=1
    ).squeeze(1)
    text_token_probs = np.array(text_token_probs)

    # heads * tokens * frames
    weights = mx.stack(
        [cross_qk[_l.item()][0, _h.item()] for _l, _h in model.alignment_heads]
    )
    weights = weights[:, :, : num_frames // 2]
    weights = mx.softmax(weights * qk_scale, axis=-1)
    mean = mx.mean(weights, axis=-2, keepdims=True)
    std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
    weights = (weights - mean) / std
    
    #weights = median_filter(np.array(weights), medfilt_width)
    weights = median_filter(mx.array(weights), medfilt_width)
    
    

    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: mx.array,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=timing.probability,
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words
