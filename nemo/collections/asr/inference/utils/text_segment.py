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


from typing import List, Set, Union

from nemo.collections.asr.inference.utils.constants import (
    BIG_EPSILON,
    DEFAULT_SEMIOTIC_CLASS,
    SEP_REPLACEABLE_PUNCTUATION,
)


def validate_init_params(
    text: str, start: float, end: float, conf: float, semiotic_class: str = None, strict: bool = False
) -> None:
    """Validate initialization parameters."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(start, (int, float)):
        raise TypeError(f"start must be numeric, got {type(start).__name__}")
    if not isinstance(end, (int, float)):
        raise TypeError(f"end must be numeric, got {type(end).__name__}")
    if not isinstance(conf, (int, float)):
        raise TypeError(f"conf must be numeric, got {type(conf).__name__}")

    if semiotic_class is not None and not isinstance(semiotic_class, str):
        raise TypeError(f"semiotic_class must be a string, got {type(semiotic_class).__name__}")

    if strict:
        if start >= end:
            raise ValueError(f"start time ({start}) must be less than end time ({end})")
        if conf < 0 or conf > 1:
            raise ValueError(f"confidence ({conf}) must be between 0 and 1")


class TextSegment:
    __slots__ = ['_text', '_start', '_end', '_conf']

    def __init__(self, text: str, start: float, end: float, conf: float) -> None:
        """
        Initialize a TextSegment instance.

        Args:
            text: The content of the text segment
            start: Start time in seconds
            end: End time in seconds
            conf: Confidence score [0.0, 1.0]
        Raises:
            ValueError: If start >= end or if confidence is negative
            TypeError: If text is not a string
        """
        validate_init_params(text, start, end, conf, strict=True)

        self._text = text
        self._start = start
        self._end = end
        self._conf = conf

    @property
    def text(self) -> str:
        """The content of the text segment."""
        return self._text

    @property
    def start(self) -> float:
        """Start time of the text segment in seconds."""
        return self._start

    @property
    def end(self) -> float:
        """End time of the text segment in seconds."""
        return self._end

    @property
    def duration(self) -> float:
        """Duration of the text segment in seconds."""
        return self._end - self._start

    @property
    def conf(self) -> float:
        """Confidence score of the text segment."""
        return self._conf

    @text.setter
    def text(self, value: str) -> None:
        """Set the content of the text segment."""
        if not isinstance(value, str):
            raise TypeError(f"text must be a string, got {type(value).__name__}")
        self._text = value

    @start.setter
    def start(self, value: float) -> None:
        """Set the start time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"start time must be numeric, got {type(value).__name__}")
        self._start = value

    @end.setter
    def end(self, value: float) -> None:
        """Set the end time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"end must be numeric, got {type(value).__name__}")
        self._end = value

    @conf.setter
    def conf(self, value: float) -> None:
        """Set the confidence score."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"conf must be numeric, got {type(value).__name__}")
        if value < 0 or value > 1:
            raise ValueError(f"confidence ({value}) must be between 0 and 1")
        self._conf = value

    def copy(self) -> 'TextSegment':
        """
        Create a deep copy of this TextSegment instance.

        Returns:
            A new TextSegment instance with identical properties
        """
        new = TextSegment(text=self.text, start=self.start, end=self.end, conf=self.conf)
        return new

    def capitalize(self) -> None:
        """Capitalize first letter of the text segment."""
        self._text = self._text.capitalize()

    def with_normalized_text(self, punct_marks: Set[str], sep: str = "") -> 'TextSegment':
        """
        Create a new TextSegment with normalized text (punctuation removed/replaced and lowercased).

        Args:
            punct_marks: Set of punctuation marks to process
            sep: Separator to replace certain punctuation marks

        Returns:
            New TextSegment instance with normalized text
        """
        replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks}
        trans_table = str.maketrans(replace_map)
        normalized_text = self.text.translate(trans_table).lower()

        # Return new instance instead of modifying in place
        obj_copy = self.copy()
        obj_copy.text = normalized_text
        return obj_copy

    def normalize_text_inplace(self, punct_marks: Set[str], sep: str = "") -> None:
        """
        Normalize text in place (punctuation removed/replaced and lowercased).

        Args:
            punct_marks: Set of punctuation marks to process
            sep: Separator to replace certain punctuation marks

        Note:
            This method modifies the current instance. Consider using
            with_normalized_text() for a functional approach.
        """
        replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks}
        trans_table = str.maketrans(replace_map)
        self.text = self.text.translate(trans_table).lower()

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another TextSegment instance.

        Args:
            other: Another object to compare with

        Returns:
            True if both instances represent the same text segment
        """
        if not isinstance(other, TextSegment):
            raise NotImplementedError(f"Cannot compare TextSegment with {type(other)}")

        return (
            self.text == other.text
            and abs(self.start - other.start) < BIG_EPSILON
            and abs(self.end - other.end) < BIG_EPSILON
            and abs(self.conf - other.conf) < BIG_EPSILON
        )

    def to_dict(self) -> dict:
        """
        Convert the TextSegment to a JSON-compatible dictionary.
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "conf": self.conf,
        }


class Word(TextSegment):
    __slots__ = ['_text', '_start', '_end', '_conf', '_semiotic_class']

    def __init__(
        self, text: str, start: float, end: float, conf: float, semiotic_class: str = DEFAULT_SEMIOTIC_CLASS
    ) -> None:
        """
        Initialize a Word instance.

        Args:
            text: The text content of the word
            start: Start time in seconds
            end: End time in seconds
            conf: Confidence score [0.0, 1.0]
            semiotic_class: Semiotic class of the word

        Raises:
            ValueError: If start >= end or if confidence is negative
            TypeError: If text is not a string
        """
        validate_init_params(text, start, end, conf, semiotic_class, strict=True)
        super().__init__(text, start, end, conf)
        self._semiotic_class = semiotic_class

    @property
    def semiotic_class(self) -> str:
        """Semiotic class of the word."""
        return self._semiotic_class

    @semiotic_class.setter
    def semiotic_class(self, value: str) -> None:
        """Set the semiotic class."""
        if not isinstance(value, str):
            raise TypeError(f"semiotic_class must be a string, got {type(value).__name__}")
        self._semiotic_class = value

    def copy(self) -> 'Word':
        """
        Create a deep copy of this Word instance.

        Returns:
            A new Word instance with identical properties
        """
        new = Word(text=self.text, start=self.start, end=self.end, conf=self.conf, semiotic_class=self.semiotic_class)
        return new

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another Word instance.

        Args:
            other: Another object to compare with

        Returns:
            True if both instances represent the same word
        """
        if not isinstance(other, Word):
            raise NotImplementedError(f"Cannot compare Word with {type(other)}")

        return super().__eq__(other) and self.semiotic_class == other.semiotic_class

    def to_dict(self) -> dict:
        """
        Convert the Word to a JSON-compatible dictionary.
        """
        return super().to_dict() | {"semiotic_class": self.semiotic_class}


def join_segments(segments: List[List[TextSegment]], sep: str) -> List[str]:
    """
    Join the text segments to form transcriptions.

    Args:
        segments: List of text segment sequences to join
        sep: Separator to use when joining text segments

    Returns:
        List of transcriptions, one for each text segment sequence
    """
    return [sep.join([s.text for s in items]) for items in segments]


def normalize_segments_inplace(
    segments: Union[List[TextSegment], List[List[TextSegment]]], punct_marks: Set[str], sep: str = ' '
) -> None:
    """
    Normalize text in text segments by removing punctuation and converting to lowercase.

    This function modifies the text segments in-place by calling normalize_text_inplace
    on each TextSegment object. It handles both flat lists of text segments and nested lists.

    Args:
        segments: List of TextSegment objects or list of lists of TextSegment objects
        punct_marks: Set of punctuation marks to be processed
        sep: Separator to replace certain punctuation marks (default: ' ')

    Note:
        This function modifies the input text segments in-place. The original text
        content of the text segments will be permanently changed.
    """
    for item in segments:
        if isinstance(item, list):
            for segment in item:
                segment.normalize_text_inplace(punct_marks, sep)
        elif isinstance(item, TextSegment):
            item.normalize_text_inplace(punct_marks, sep)
        else:
            raise ValueError(f"Invalid item type: {type(item)}. Expected `TextSegment` or `List[TextSegment]`.")
