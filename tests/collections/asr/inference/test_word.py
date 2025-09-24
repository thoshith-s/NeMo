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


import pytest
from nemo.collections.asr.inference.utils.word import Word, join_words, normalize_words_inplace


class TestWord:

    @pytest.mark.unit
    def test_word_equality(self):
        word1 = Word('hello', 0, 1, 0.5)
        word2 = Word('hello', 0, 1, 0.5)
        assert word1 == word2

        word3 = word1.copy()
        assert word1 == word3

        with pytest.raises(NotImplementedError):
            word1 == 'hello'

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("Hello!", "hello"), ("HeLLo!", "hello")])
    def test_normalize_text_inplace(self, text, expected_text):
        word = Word(text, 0, 1, 0.5)
        word.normalize_text_inplace(punct_marks='!', sep=' ')
        assert word.text == expected_text

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("Hello!", "hello"), ("HeLLo!", "hello")])
    def test_with_normalized_text(self, text, expected_text):
        word = Word(text, 0, 1, 0.5)
        word_copy = word.with_normalized_text(punct_marks='!', sep=' ')
        assert word_copy.text == expected_text
        assert word.text == text

    @pytest.mark.unit
    def test_join_words(self):
        words = [
            [Word('hello', 0, 1, 0.5), Word('world', 1, 2, 0.5)],
            [Word('how', 2, 3, 0.5), Word('are', 3, 4, 0.5), Word('you', 4, 5, 0.5)],
        ]
        transcriptions = join_words(words, sep=' ')
        assert transcriptions == ['hello world', 'how are you']

    @pytest.mark.unit
    def test_normalize_words_inplace(self):
        words = [Word('Hello!', 0, 1, 0.5), Word('world?', 1, 2, 0.5)]
        normalize_words_inplace(words, punct_marks=set("!?"), sep=' ')
        assert words[0].text == 'hello'
        assert words[1].text == 'world'

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("hello", "Hello"), ("World!", "World!")])
    def test_capitalize(self, text, expected_text):
        word = Word(text, 0, 1, 0.5)
        word.capitalize()
        assert word.text == expected_text
