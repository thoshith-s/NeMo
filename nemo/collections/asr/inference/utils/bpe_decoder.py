# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import lru_cache
from typing import Callable, List, Set, Tuple

import numpy as np

from nemo.collections.asr.inference.utils.constants import (
    POST_WORD_PUNCTUATION,
    ROUND_PRECISION,
    SENTENCEPIECE_UNDERSCORE,
)
from nemo.collections.asr.inference.utils.word import Word
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class BPEDecoder:
    """
    BPEDecoder class for decoding BPE (Byte Pair Encoding) tokens into words with associated timestamps and confidence scores
    """

    def __init__(
        self,
        vocabulary: List[str],
        tokenizer: TokenizerSpec,
        confidence_aggregator: Callable,
        asr_supported_puncts: Set,
        word_boundary_tolerance: float,
        token_duration_in_secs: float,
    ):

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.confidence_aggregator = confidence_aggregator
        self.asr_supported_puncts = asr_supported_puncts
        self.punct_marks_with_underscore = asr_supported_puncts.union({SENTENCEPIECE_UNDERSCORE})
        self.word_boundary_tolerance = word_boundary_tolerance
        self.token_duration_in_secs = token_duration_in_secs
        self.start_of_word_cache = {
            token_id: token.startswith(SENTENCEPIECE_UNDERSCORE) for token_id, token in enumerate(self.vocabulary)
        }
        self.punct_cache = {
            token_id: (token in self.asr_supported_puncts, token in self.punct_marks_with_underscore)
            for token_id, token in enumerate(self.vocabulary)
        }

    @lru_cache(maxsize=10000)
    def cached_tokenizer(self, tokens_slice: Tuple[int]) -> str:
        """
        Cached tokenizer to avoid repeated calls to the tokenizer.
        Args:
            tokens_slice (tuple): Tuple of token indices to be detokenized.
        Returns:
            str: Detokenized text.
        """
        word_text = self.tokenizer.ids_to_text(list(tokens_slice)).strip()
        return word_text

    def bpe_decode(self, tokens: List, timesteps: List, confidences: List) -> Tuple[List[Word], bool]:
        """
        Decodes BPE tokens into words with timestamps and confidence scores.
        Args:
            tokens (list): List of token indices.
            timesteps (list): List of token timesteps.
            confidences (list): List of token confidence scores.
        Returns:
            list: List of decoded words with text, start time, end time, and confidence score.
            merge_first_word: True if the first word should be merged with the last word stored in the state
        """
        n_tokens = len(tokens)

        if n_tokens != len(timesteps) or n_tokens != len(confidences):
            raise ValueError("tokens, timesteps and confidences must have the same length")

        if n_tokens == 0:
            return [], False

        # Group tokens into words
        is_start_mask = np.fromiter((self.start_of_word_cache[tok] for tok in tokens), dtype=np.int32)
        word_ids = np.cumsum(is_start_mask)

        start_indices = np.nonzero(np.diff(word_ids, prepend=word_ids[0] - 1))[0]
        end_indices = np.append(start_indices[1:], n_tokens)

        decoded_words, prev_word_end = [], None

        # If the first word is the start of a word, we need to merge it with the last word stored in the state
        merge_first_word = not bool(is_start_mask[0])

        for start_idx, end_idx in zip(start_indices, end_indices):

            tokens_slice = tokens[start_idx:end_idx]
            time_slice = timesteps[start_idx:end_idx]
            conf_slice = confidences[start_idx:end_idx]

            word_text = self.cached_tokenizer(tuple(tokens_slice))

            # Ignore empty text
            if not word_text:
                continue

            # Append the post word punctuation to the previous word
            if word_text in POST_WORD_PUNCTUATION and len(decoded_words) > 0:
                prev_word = decoded_words[-1]
                prev_word.text += word_text
                continue

            # Refine timestamps
            word_start_tms, word_end_tms = self.refine_word_timestamp(
                current_word_tokens=tokens_slice,
                current_word_timesteps=time_slice,
                next_word_start_timestep=timesteps[end_idx] if end_idx < n_tokens else None,
                is_next_word_start_of_word=(self.start_of_word_cache[tokens[end_idx]] if end_idx < n_tokens else None),
                prev_word_end=prev_word_end,
            )
            prev_word_end = word_end_tms

            # Aggregate confidence
            word_conf = self.confidence_aggregator(conf_slice)

            # Convert token timestamps to seconds
            start_sec = round(word_start_tms * self.token_duration_in_secs, ROUND_PRECISION)
            end_sec = round(word_end_tms * self.token_duration_in_secs, ROUND_PRECISION)

            decoded_words.append(Word(text=word_text, start=start_sec, end=end_sec, conf=word_conf))

        return decoded_words, merge_first_word

    def refine_word_timestamp(
        self,
        current_word_tokens: List[int],
        current_word_timesteps: List[float],
        next_word_start_timestep: float | None,
        is_next_word_start_of_word: bool | None,
        prev_word_end: float | None,
    ) -> Tuple[float, float]:
        """
        Refines the word timestamp based on the current word tokens, timestamps, and the next word start timestamp.
        Args:
            current_word_tokens (list): List of token indices.
            current_word_timesteps (list): List of token timestamps.
            next_word_start_timestep (float): The start timestamp of the next word.
            is_next_word_start_of_word (bool): True if the next word is the start of a word.
            prev_word_end (float): The end timestamp of the previous word.
        Returns:
            tuple (float, float): The refined start and end timestamps.
        """

        start, end = current_word_timesteps[0], current_word_timesteps[-1]

        # --- Correct the start timestamp if the first token is underscore or punctuation ---
        first_token = current_word_tokens[0]
        if self.punct_cache[first_token][1]:
            start = next(
                (
                    tms
                    for tms, token in zip(current_word_timesteps, current_word_tokens)
                    if not self.punct_cache[token][1]
                ),
                start,
            )

        # --- Correct the end timestamp if the last token is punctuation ---
        last_token = current_word_tokens[-1]
        if self.punct_cache[last_token][0]:
            end = next(
                (
                    current_word_timesteps[i]
                    for i in reversed(range(len(current_word_tokens)))
                    if not self.punct_cache[current_word_tokens[i]][0]
                ),
                end,
            )

        # --- If the next word is close to the end of the current word, merge timestamps ---
        if next_word_start_timestep is not None and is_next_word_start_of_word:
            if next_word_start_timestep - end <= self.word_boundary_tolerance:
                end = next_word_start_timestep

        delta = 0
        if prev_word_end is not None:
            if prev_word_end > start:
                delta = prev_word_end - start

        start = start + delta
        end = end + delta
        return start, end + (1 if start == end else 0)
