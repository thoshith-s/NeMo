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

import re
from abc import abstractmethod
from typing import Any, Iterable

from nemo.collections.asr.inference.asr.asr_inference import ASRInference
from nemo.collections.asr.inference.recognizers.recognizer_interface import RecognizerInterface
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.asr.inference.utils.recognizer_utils import get_leading_punctuation_regex_pattern
from nemo.collections.asr.inference.utils.text_segment import TextSegment
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class RecognizerOutput:
    """
    Class to store the output of the recognizer.
    """

    def __init__(self, texts: list[str] | None = None, segments: list[list[TextSegment]] | None = None):
        if texts is None and segments is None:
            raise ValueError("At least one of the 'texts' or 'segments' should be provided.")
        self.texts = texts
        self.segments = segments


class BaseRecognizer(RecognizerInterface):
    """
    Base class for all recognizers.
    """

    def __init__(self):
        """Initialize state pool to store the state for each stream"""
        self._state_pool: dict[int, Any] = {}

    def get_state(self, stream_id: int) -> Any:
        """Retrieve state for a given stream ID."""
        return self._state_pool.get(stream_id, None)

    def get_states(self, stream_ids: Iterable[int]) -> list[Any]:
        """Retrieve states for a list of stream IDs."""
        return [self.get_state(stream_id) for stream_id in stream_ids]

    def delete_state(self, stream_id: int) -> None:
        """Delete the state from the state pool."""
        if stream_id in self._state_pool:
            del self._state_pool[stream_id]

    def delete_states(self, stream_ids: Iterable[int]) -> None:
        """Delete states for a list of stream IDs."""
        for stream_id in stream_ids:
            self.delete_state(stream_id)

    def init_state(self, stream_id: int, options: ASRRequestOptions) -> Any:
        """Initialize the state of the stream"""
        if stream_id not in self._state_pool:
            state = self.create_state(options)
            self._state_pool[stream_id] = state
        return self._state_pool[stream_id]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self._state_pool.clear()

    def open_session(self) -> None:
        """Start a new session by resetting the internal state pool"""
        self.reset_session()

    def close_session(self) -> None:
        """Close the session by resetting the internal state pool"""
        self.reset_session()

    @abstractmethod
    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """Transcribe a step for frames"""
        pass

    @abstractmethod
    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        pass

    @abstractmethod
    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Return the request generator."""
        pass

    @abstractmethod
    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        pass

    def transcribe_step(self, requests: list[Request]) -> None:
        """Transcribe a step"""
        if isinstance(requests[0], Frame):
            self.transcribe_step_for_frames(frames=requests)
        elif isinstance(requests[0], FeatureBuffer):
            self.transcribe_step_for_feature_buffers(fbuffers=requests)
        else:
            raise ValueError(f"Invalid request type: {type(requests[0])}")

    def copy_asr_model_attributes(self, asr_model: ASRInference) -> None:
        """
        Copy the attributes from the ASR model to the recognizer
        Args:
            asr_model (ASRInference): ASR model to copy the attributes from.
        """
        self.asr_model = asr_model
        self.tokenizer = asr_model.tokenizer
        self.device = asr_model.device
        self.supports_punctuation = asr_model.supports_punctuation()
        self.asr_supported_puncts = asr_model.supported_punctuation()
        self.leading_regex_pattern = get_leading_punctuation_regex_pattern(self.asr_supported_puncts)
        self.blank_id = asr_model.get_blank_id()
        self.vocabulary = asr_model.get_vocabulary()
        self.sep = asr_model.word_separator
        self.segment_separators = asr_model.segment_separators
        self.underscore_id = asr_model.underscore_id
        self.punctuation_ids = asr_model.punctuation_ids
        self.language_token_ids = asr_model.language_token_ids
        self.preprocessor, self.preprocessor_config = asr_model.create_preprocessor()
        self.subsampling_factor = asr_model.get_subsampling_factor()
        self.window_stride = asr_model.get_window_stride()
        self.model_stride_in_secs = asr_model.get_model_stride(in_secs=True)
        self.model_stride_in_milliseconds = asr_model.get_model_stride(in_milliseconds=True)

    def update_partial_transcript(
        self, requests: list[Request], tokenizer: TokenizerSpec, leading_regex_pattern: str
    ) -> None:
        """
        Update partial transcript from the state.
        Args:
            requests (list[Request]): List of Request objects.
            tokenizer (TokenizerSpec): Used to convert tokens into text
            leading_regex_pattern (str): Regex pattern for the punctuation marks.
        """
        for request in requests:
            state = self.get_state(request.stream_id)
            # state tokens represent all tokens accumulated since the EOU
            # incomplete segment tokens are the remaining tokens on the right side of the buffer after EOU
            all_tokens = state.tokens + state.incomplete_segment_tokens
            if len(all_tokens) > 0:
                pt_string = tokenizer.ids_to_text(all_tokens)
                if leading_regex_pattern:
                    pt_string = re.sub(leading_regex_pattern, r'\1', pt_string)
                state.partial_transcript = pt_string
            else:
                state.partial_transcript = ""

    def run(
        self,
        audio_filepaths: list[str],
        options: list[ASRRequestOptions] | None = None,
        progress_bar: ProgressBar | None = None,
    ) -> RecognizerOutput:
        """
        Orchestrates reading from audio_filepaths in a streaming manner,
        transcribes them, and packs the results into a RecognizerOutput.
        Args:
            audio_filepaths (list[str]): List of audio filepaths to transcribe.
            options (list[ASRRequestOptions] | None): List of RequestOptions for each stream.
            progress_bar (ProgressBar | None): Progress bar to show the progress. Default is None.
        Returns:
            RecognizerOutput: A dataclass containing transcriptions and segments.
        """
        if progress_bar is not None and not isinstance(progress_bar, ProgressBar):
            raise ValueError("progress_bar must be an instance of ProgressBar.")

        if options is None:
            # Use default options if not provided
            options = [ASRRequestOptions() for _ in audio_filepaths]

        if len(options) != len(audio_filepaths):
            raise ValueError("options must be the same length as audio_filepaths")

        request_generator = self.get_request_generator()
        request_generator.set_audio_filepaths(audio_filepaths, options)
        request_generator.set_progress_bar(progress_bar)

        self.open_session()
        for requests in request_generator:
            for request in requests:
                if request.is_first:
                    self.init_state(request.stream_id, request.options)
            self.transcribe_step(requests)
        output = self.pack_output()
        self.close_session()
        return output

    def pack_output(self) -> RecognizerOutput:
        """Pack the output from the internal state pool."""
        texts, segments = [], []
        for stream_id in sorted(self._state_pool):
            state = self.get_state(stream_id)
            if state.options.is_word_level_output():
                attr_name = "itn_words" if state.options.enable_itn else "pnc_words"
                state_segments = getattr(state, attr_name)
                state_text = self.get_sep().join(word.text for word in state_segments)
            else:
                # Segment-level output branch
                state_segments = getattr(state, "segments")
                state_text = self.get_sep().join(segment.text for segment in state_segments)
            texts.append(state_text)
            segments.append(state_segments)

        return RecognizerOutput(texts=texts, segments=segments)
