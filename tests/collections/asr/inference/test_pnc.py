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
import torch

from nemo.collections.asr.inference.pnc.punctuation_capitalizer import PunctuationCapitalizer, Word


@pytest.fixture(scope="module")
def punctuation_en_distilbert_model():
    return PunctuationCapitalizer(
        model_name="punctuation_en_distilbert", device="cuda" if torch.cuda.is_available() else "cpu"
    )


class TestPunctuationCapitalizer:

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_punctuate_words(self, punctuation_en_distilbert_model):
        words = [
            Word("hello", 0, 1, 0.5),
            Word("world", 1, 2, 0.5),
            Word("how", 2, 3, 0.5),
            Word("are", 3, 4, 0.5),
            Word("you", 4, 5, 0.5),
        ]

        pnc_words = punctuation_en_distilbert_model.add_punctuation_capitalization_to_words(
            [words], params={}, sep=" "
        )[0]
        assert pnc_words[0].text.startswith("Hello")
        assert pnc_words[-1].text.endswith("you?")
        assert isinstance(pnc_words[0], Word)
        assert len(pnc_words) == len(words)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_punctuation_capitalizer(self, punctuation_en_distilbert_model):
        text = "hello world how are you"
        result = punctuation_en_distilbert_model.add_punctuation_capitalization_list([text], params={})
        assert result[0].startswith("Hello")
        assert result[0].endswith("you?")
        assert isinstance(result[0], str)
        assert len(result) == 1
        assert isinstance(result, list)
        assert len(result[0].split()) == len(text.split())
