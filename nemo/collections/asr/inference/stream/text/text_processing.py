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

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig

from nemo.collections.asr.inference.stream.state.state import StreamingState
from nemo.collections.asr.inference.utils.constants import POST_WORD_PUNCTUATION
from nemo.collections.asr.inference.utils.recognizer_utils import (
    apply_regex_substitution,
    get_leading_punctuation_regex_pattern,
    get_repeated_punctuation_regex_pattern,
)
from nemo.collections.asr.inference.utils.text_segment import Word, normalize_segments_inplace
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.pnc.punctuation_capitalizer import PunctuationCapitalizer


class StreamingTextPostprocessor:
    """
    A streaming text post-processing module that applies punctuation & capitalization (PnC) and
    inverse text normalization (ITN) to ASR transcriptions in real-time.

    This class supports configurable pipelines where PnC and ITN can be enabled/disabled dynamically.
    It ensures that the final output adheres to proper punctuation, capitalization, and normalized text
    formatting by leveraging a PnC model and a WFST-based ITN model.
    """

    def __init__(
        self,
        text_postprocessor_cfg: DictConfig,
        pnc_model: Optional[PunctuationCapitalizer],
        itn_model: Optional[AlignmentPreservingInverseNormalizer],
        asr_supported_puncts: Set,
        asr_supports_punctuation: bool,
        confidence_aggregator: Callable,
        sep: str,
        segment_separators: List[str],
        automatic_punctuation: bool = False,
        verbatim_transcripts: bool = True,
    ):
        """
        Initialize the streaming text postprocessor.

        Args:
            text_postprocessor_cfg: Configuration object for text post-processing settings.
            pnc_model: Model for punctuation and capitalization (PnC).
            itn_model: Model for inverse text normalization (ITN).
            asr_supported_puncts: Set of punctuation marks recognized by the ASR model.
            asr_supports_punctuation: Boolean indicating if the ASR model outputs punctuation.
            confidence_aggregator: Function for aggregating confidence scores.
            sep: String separator used in ASR output processing.
            segment_separators: List of characters used as segment boundaries.
        """

        self.pnc_model = pnc_model
        self.pnc_enabled = False
        if automatic_punctuation:
            self.pnc_enabled = pnc_model is not None or asr_supports_punctuation

        self.force_to_use_pnc_model = text_postprocessor_cfg.force_to_use_pnc_model
        if self.pnc_model is None:
            self.force_to_use_pnc_model = False

        self.supports_punctuation = asr_supports_punctuation
        self.pnc_params = text_postprocessor_cfg.pnc
        self.pnc_left_padding_search_size = text_postprocessor_cfg.pnc.left_padding_search_size

        self.itn_model = itn_model
        self.itn_enabled = False
        if not verbatim_transcripts:
            self.itn_enabled = itn_model is not None

        self.itn_params = text_postprocessor_cfg.itn
        self.itn_left_padding_size = text_postprocessor_cfg.itn.left_padding_size

        self.asr_supported_puncts = asr_supported_puncts
        self.asr_supported_puncts_str = ''.join(self.asr_supported_puncts)
        self.sep = sep
        self.segment_separators = segment_separators
        self.rm_punctuation_capitalization_from_segments_fn = partial(
            normalize_segments_inplace, punct_marks=self.asr_supported_puncts, sep=self.sep
        )

        puncts_to_process = self.asr_supported_puncts
        if self.pnc_model is not None:
            puncts_to_process = puncts_to_process | self.pnc_model.supported_punctuation
        self.leading_punctuation_regex_pattern = get_leading_punctuation_regex_pattern(puncts_to_process)
        self.repeated_punctuation_regex_pattern = get_repeated_punctuation_regex_pattern(puncts_to_process)

        self.alignment_aware_itn_model = None
        if self.itn_enabled:
            from nemo.collections.asr.inference.itn.batch_inverse_normalizer import (
                BatchAlignmentPreservingInverseNormalizer,
            )

            self.alignment_aware_itn_model = BatchAlignmentPreservingInverseNormalizer(
                itn_model=self.itn_model,
                sep=self.sep,
                asr_supported_puncts=self.asr_supported_puncts,
                post_word_punctuation=POST_WORD_PUNCTUATION,
                conf_aggregate_fn=confidence_aggregator,
            )

    def is_itn_enabled(self) -> bool:
        """Check if ITN is enabled"""
        return self.itn_enabled

    def is_pnc_enabled(self) -> bool:
        """Check if PnC is enabled"""
        return self.pnc_enabled

    def is_enabled(self) -> bool:
        """Check if PnC or ITN is enabled"""
        return self.is_pnc_enabled() or self.is_itn_enabled()

    def add_pnc_to_words(self, words_list: List[List[Word]], pnc_params: Dict) -> List[List[Word]]:
        """
        Run the Punctuation and Capitalization model on the words.
        Args:
            words_list: (List[List[Word]]) List of words
            pnc_params: (Dict) Runtime parameters for the PNC model
        Returns:
            (List[List[Word]]) List of words after applying the PnC model
        """
        if self.pnc_model is None:
            logging.info("PnC model is not provided. Skipping PnC.")
            return words_list
        return self.pnc_model.add_punctuation_capitalization_to_words(words_list, pnc_params, sep=self.sep)

    def process(self, states: List[StreamingState]) -> None:
        """
        Apply PnC and ITN on the states.
        Args:
            states: (List[StreamingState]) List of StreamingState objects
        """
        word_boundary_states, segment_boundary_states = [], []
        for state in states:
            if state.options.is_word_level_output():
                word_boundary_states.append(state)
            else:
                segment_boundary_states.append(state)

        # Process states with word boundaries
        if word_boundary_states:
            self.process_states_with_word_boundaries(word_boundary_states)

        # Process states with segment boundaries
        if segment_boundary_states:
            self.process_states_with_segment_boundaries(segment_boundary_states)

        # Generate final transcript
        self.generate_final_transcript(word_boundary_states, segment_boundary_states)

    def process_states_with_segment_boundaries(self, states: List[StreamingState]) -> None:
        """
        Process states with segment boundaries.
        Args:
            states: List of StreamingState objects that have segments
        """
        states_with_text = [state for state in states if len(state.segments) > 0]
        if len(states_with_text) == 0:
            return

        # if PnC & ITN DISABLED globally, remove PnC from the words if ASR supports punctuation
        if not self.is_enabled():
            if self.supports_punctuation:
                segments = []
                for state in states_with_text:
                    for i, seg in enumerate(state.segments):
                        if not state.processed_segment_mask[i]:
                            segments.append(seg)
                            state.processed_segment_mask[i] = True
                self.rm_punctuation_capitalization_from_segments_fn(segments)
            return

        # Remove PnC from states where PnC is disabled
        for state in states_with_text:
            if (not state.options.enable_pnc) or (not self.is_pnc_enabled()):
                if self.supports_punctuation:
                    self.rm_punctuation_capitalization_from_segments_fn(state.segments)

        # Apply ITN
        if self.is_itn_enabled():  # If ITN ENABLED globally
            # collect texts
            texts = []
            for i, state in enumerate(states_with_text):
                # if ITN is disabled for this state
                if not state.options.enable_itn:
                    continue

                for j, seg in enumerate(state.segments):
                    if state.processed_segment_mask[j]:  # if the segment is already processed, skip it
                        continue
                    texts.append((i, j, seg.text))

            if len(texts) > 0:
                # apply ITN
                processed_texts = self.itn_model.inverse_normalize_list(
                    texts=[text for _, _, text in texts], params=self.itn_params
                )
                # update states with ITN-processed texts
                for (i, j, _), processed_text in zip(texts, processed_texts):
                    states_with_text[i].segments[j].text = processed_text

        # Apply PnC
        if self.is_pnc_enabled():  # If PnC ENABLED globally
            texts = []
            for i, state in enumerate(states_with_text):
                # if PnC is disabled for this state
                if not state.options.enable_pnc:
                    continue

                # if ASR does not support PnC -> process the text
                # or
                # if ASR supports PnC but we force to use the BERT-based PnC model -> process the text
                should_process = not self.supports_punctuation
                if self.supports_punctuation and self.force_to_use_pnc_model:
                    should_process = True

                if should_process:
                    for j, seg in enumerate(state.segments):
                        if state.processed_segment_mask[j]:  # if the segment is already processed, skip it
                            continue
                        texts.append((i, j, seg.text))

            if len(texts) > 0:
                # apply PnC
                processed_texts = self.pnc_model.add_punctuation_capitalization_list(
                    transcriptions=[text for _, _, text in texts], params=self.pnc_params
                )
                # update states with PnC-processed texts
                for (i, j, _), processed_text in zip(texts, processed_texts):
                    states_with_text[i].segments[j].text = processed_text

        # mark all segments as processed
        for state in states_with_text:
            if state.options.enable_pnc:
                for seg in state.segments:
                    seg.text = apply_regex_substitution(seg.text, self.leading_punctuation_regex_pattern, r'\1')
                    seg.text = apply_regex_substitution(seg.text, self.repeated_punctuation_regex_pattern, r'\1')
            state.processed_segment_mask = [True] * len(state.segments)

    def process_states_with_word_boundaries(self, states: List[StreamingState]) -> None:
        """
        Apply PnC and ITN on the states.
        Args:
            states: (List[StreamingState]) List of StreamingState objects
        """
        # Get the indices of the states that have new words to process
        indices, asr_words_list = self.prepare_asr_words(states)

        # If PnC & ITN DISABLED globally, remove PnC from the words
        # Does not matter that individual request has enabled itn or pnc
        if not self.is_enabled():
            self.handle_plain_asr_transcriptions(states, indices, asr_words_list)
            return

        # Apply Punctuation and Capitalization (PnC)
        pnc_words_list = [None] * len(asr_words_list)
        subset_to_punctuate, subset_indices = [], []
        for idx, jdx, _, _ in indices:
            if states[idx].options.enable_pnc:
                subset_to_punctuate.append(asr_words_list[jdx])
                subset_indices.append(jdx)
            else:
                # if PnC is disabled for particular states, remove PnC from the words if ASR supports punctuation
                if self.supports_punctuation:
                    self.rm_punctuation_capitalization_from_segments_fn(asr_words_list[jdx])
                pnc_words_list[jdx] = asr_words_list[jdx]

        for jdx, pnc_words in zip(subset_indices, self.apply_pnc(subset_to_punctuate)):
            pnc_words_list[jdx] = pnc_words

        # Update states with PnC results
        for idx, jdx, z, n_x in indices:
            state = states[idx]
            z = z - n_x
            if state.options.enable_pnc and n_x > 0:
                # Handle the first word of the response
                first_word: Word = pnc_words_list[jdx][-z]
                prev_word: Word = pnc_words_list[jdx][-z - 1] if abs(-z - 1) <= len(pnc_words_list[jdx]) else None
                if prev_word is not None:
                    if prev_word.text[-1] in POST_WORD_PUNCTUATION:
                        first_word.capitalize()
            states[idx].pnc_words[-z:] = pnc_words_list[jdx][-z:]

        # If ITN is disabled globally, do nothing
        if not self.itn_enabled:
            return

        # Apply Inverse Text Normalization (ITN)
        self.apply_itn(states, indices)
        self.realign_punctuated_words(states, indices)

    def realign_punctuated_words(self, states: List[StreamingState], indices: List[tuple]) -> None:
        """
        Realign punctuation and capitalization after applying ITN.
        Ensures that capitalization and punctuation marks from the original ASR output
        are properly reflected in the final ITN-processed text.

        Args:
            states: List of StreamingState objects to be updated.
            indices: Indices of words within states that need realignment.
        """
        for idx, _, z, n_x in indices:
            state = states[idx]
            if not state.options.enable_itn:
                continue

            z = z - n_x
            z_idx = len(state.words) - z

            itn_idx = len(state.itn_words)
            for sids, _, _ in reversed(state.word_alignment):
                st, et = sids[0], sids[-1]
                itn_idx -= 1
                if st < z_idx and et < z_idx:
                    break

                last_char = state.pnc_words[et].text[-1]
                first_char = state.pnc_words[st].text[0]

                itn_word_orig = state.itn_words[itn_idx]
                itn_word_copy = itn_word_orig.copy()
                itn_word_text = itn_word_copy.text.lower()

                # preserve the first char capitalization
                first_word = state.pnc_words[st].copy()
                first_char_is_upper = first_word.text[0].isupper()
                first_word.normalize_text_inplace(self.asr_supported_puncts, self.sep)
                if first_char_is_upper and itn_word_text.startswith(first_word.text):
                    itn_word_orig.capitalize()

                # preserve the last punctuation mark
                if last_char in self.asr_supported_puncts:
                    itn_word_orig.text = itn_word_orig.text.rstrip(self.asr_supported_puncts_str) + last_char

                # preserve the first punctuation mark
                if first_char in self.asr_supported_puncts:
                    itn_word_orig.text = first_char + itn_word_orig.text.lstrip(self.asr_supported_puncts_str)

                state.itn_words[itn_idx] = itn_word_orig

    def prepare_asr_words(self, states: List[StreamingState]) -> Tuple[List[Tuple], List[List[Word]]]:
        """
        Prepare words for applying PnC and ITN.
        Find the indices of the states that have words to process.
        Calculate the number of words to look back for PnC.
        Args:
            states: (List[StreamingState]) List of StreamingState objects
        Returns:
            List[tuple]: List of indices of the states that have words to process
            List[List[Word]]: Batch of words to process
        """
        indices, asr_words_list = [], []

        jdx = 0
        for idx, state in enumerate(states):
            if (n_not_punctuated_words := len(state.words) - len(state.pnc_words)) == 0:
                continue

            n_word_lookback = 0
            if self.pnc_enabled:
                n_word_lookback = self.calculate_pnc_lookback(state.pnc_words)

            # if no lookback words is found, use the context before response
            n_additional_context = 0
            if n_word_lookback == 0 and len(state.context_before_response) > 0:
                n_additional_context = min(len(state.context_before_response), self.pnc_left_padding_search_size)

            # Prepare words with additional context
            if n_additional_context == 0:
                z = n_not_punctuated_words + n_word_lookback
                words_list = [word.copy() for word in state.words[-z:]]
            else:
                words_list = [
                    word.copy()
                    for word in state.context_before_response[-n_additional_context:]
                    + state.words[-n_not_punctuated_words:]
                ]

            asr_words_list.append(words_list)
            state.pnc_words.extend([None] * n_not_punctuated_words)
            indices.append((idx, jdx, len(words_list), n_additional_context))
            jdx += 1

        return indices, asr_words_list

    def calculate_pnc_lookback(self, pnc_words: List[Word]) -> int:
        """
        Calculate the number of words to look back for PnC.
        Args:
            pnc_words: (List[Word]) List of punctuated words
        Returns:
            (int) Number of words to look back for PnC
        """
        if len(pnc_words) == 0:
            return 0

        lookback = 0
        for ix, word in enumerate(reversed(pnc_words), start=1):
            if word.text and word.text[-1] in self.segment_separators:
                lookback = ix - 1
            if ix == self.pnc_left_padding_search_size:
                break
        return lookback

    def handle_plain_asr_transcriptions(
        self, states: List[StreamingState], indices: List[Tuple], asr_words_list: List[List[Word]]
    ) -> None:
        """
        Handle scenarios where PnC and ITN are disabled.
        In such cases, remove Punctuation and Capitalization from the words.
        """
        if self.supports_punctuation:
            self.rm_punctuation_capitalization_from_segments_fn(asr_words_list)

        for idx, jdx, z, n_x in indices:
            z = z - n_x
            states[idx].pnc_words[-z:] = asr_words_list[jdx][-z:]

    def apply_pnc(self, asr_words_list: List[List[Word]]) -> List[List[Word]]:
        """
        Apply Punctuation and Capitalization (PnC) on the words.
        If PnC is disabled, remove Punctuation and Capitalization from the words.
        If force_to_use_pnc_model is enabled, force to use the BERT-based PnC model.
        Args:
            asr_words_list: (List[List[Word]]) List of words
        Returns:
            (List[List[Word]]) List of words after applying the PnC model
        """
        if self.pnc_enabled:
            if self.force_to_use_pnc_model:
                logging.info("Forcing to use BERT-based PnC model.")
                if self.supports_punctuation:
                    self.rm_punctuation_capitalization_from_segments_fn(asr_words_list)
                return self.add_pnc_to_words(asr_words_list, self.pnc_params)

            if not self.supports_punctuation:
                return self.add_pnc_to_words(asr_words_list, self.pnc_params)

        elif self.supports_punctuation:
            self.rm_punctuation_capitalization_from_segments_fn(asr_words_list)
        return asr_words_list

    def apply_itn(self, states: List[StreamingState], indices: List[Tuple]) -> None:
        """
        Apply Inverse Text Normalization (ITN) on the states.
        Calculates the lookback for ITN and updates the states with the ITN results.
        Args:
            states: (List[StreamingState]) List of StreamingState objects
            indices: (List[Tuple]) List of indices of the states that have words to process
        """
        itn_indices, asr_words_list, pnc_words_list = [], [], []
        jdx = 0
        for state_idx, _, _, _ in indices:
            state = states[state_idx]
            if not state.options.enable_itn:
                continue
            s, t, cut_point = self.calculate_itn_lookback(state)
            asr_words_list.append([word.copy() for word in state.words[s:]])
            pnc_words_list.append([word.copy() for word in state.pnc_words[s:]])
            itn_indices.append((state_idx, jdx, s, t, cut_point))
            jdx += 1
        output = self.alignment_aware_itn_model(asr_words_list, pnc_words_list, self.itn_params, return_alignment=True)
        self.update_itn_words(states, output, itn_indices)

    def calculate_itn_lookback(self, state: StreamingState) -> Tuple[int, int, int]:
        """
        Calculate the lookback for ITN.
        Args:
            state: (StreamingState) StreamingState object
        Returns:
            Start index (int): Start index of the source (non itn-ed) words
            Target index (int): Start index of the target (itn-ed) words
            Cut point (int): Index to cut the source words
        """
        s, t, cut_point = 0, 0, len(state.itn_words)
        word_alignment = list(reversed(state.word_alignment))
        for idx, (sidx, tidx, _) in enumerate(word_alignment, start=1):
            s, t = sidx[0], tidx[0]
            state.word_alignment.pop()
            cut_point -= 1
            if idx == self.itn_left_padding_size:
                break
        return s, t, cut_point

    @staticmethod
    def update_itn_words(states: List[StreamingState], output: List[Tuple], indices: List[Tuple]) -> None:
        """
        Update the states with the ITN results.
        Update word_alignment and itn_words.
        """
        for state_idx, jdx, s, t, cut_point in indices:
            state = states[state_idx]
            spans, alignment = output[jdx]
            for sidx, tidx, sclass in alignment:
                sidx = [k + s for k in sidx]
                tidx = [k + t for k in tidx]
                state.word_alignment.append((sidx, tidx, sclass))

            state.itn_words = state.itn_words[:cut_point] + spans
            assert len(state.word_alignment) == len(state.itn_words)

    def generate_final_transcript(
        self, word_boundary_states: List[StreamingState], segment_boundary_states: List[StreamingState]
    ) -> None:
        """
        Generate final transcript based on enabled features and word count.
        Args:
            word_boundary_states (List[StreamingState]): The streaming state containing words
            segment_boundary_states (List[StreamingState]): The streaming state containing segments
        """
        # Generate final transcript for word boundary states
        for state in word_boundary_states:
            attr_name = "itn_words" if state.options.enable_itn else "pnc_words"
            words = getattr(state, attr_name)
            for word in words:
                state.final_words.append(word.copy())
                state.final_transcript += word.text + self.sep
            state.final_transcript = state.final_transcript.rstrip(self.sep)

        # Generate final transcript for segment boundary states
        for state in segment_boundary_states:
            for segment in state.segments:
                state.final_segments.append(segment.copy())
                state.final_transcript += segment.text + self.sep
            state.final_transcript = state.final_transcript.rstrip(self.sep)
