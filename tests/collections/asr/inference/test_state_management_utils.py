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

from nemo.collections.asr.inference.utils.state_management_utils import (
    detect_overlap,
    find_max_overlap,
    merge_timesteps,
    merge_word_tail,
)
from nemo.collections.asr.inference.utils.word import Word


class TestStateManagementUtils:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "timesteps1, timesteps2, expected_merged_timesteps",
        [
            ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
            ([0, 1, 2, 3], [], [0, 1, 2, 3]),
            ([], [4, 5, 6, 7], [4, 5, 6, 7]),
            ([-1, 0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]),
            ([-3, 1, 2, 3], [], [0, 4, 5, 6]),
            ([], [-3, 1, 2, 3], [0, 4, 5, 6]),
        ],
    )
    def test_merge_timesteps(self, timesteps1, timesteps2, expected_merged_timesteps):
        merged_timesteps = merge_timesteps(timesteps1, timesteps2)
        assert merged_timesteps == expected_merged_timesteps

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "state_tokens, new_tokens, limit, expected_max_overlap",
        [
            ([0, 1, 2, 3], [2, 3, 4, 5], 4, 2),
            ([0, 2, 3, 4], [2, 3, 4, 5], 4, 3),
            ([0, 0, 0, 1], [2, 3, 4, 5], 4, 0),
        ],
    )
    def test_find_max_overlap(self, state_tokens, new_tokens, limit, expected_max_overlap):
        max_overlap = find_max_overlap(state_tokens, new_tokens, limit)
        assert max_overlap == expected_max_overlap

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "state_tokens, state_timesteps, new_tokens, new_timesteps, expected_overlap",
        [
            ([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0], [2, 3, 4, 5], [2.0, 3.0, 4.0, 5.0], 2),
            ([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0], [2, 3, 4, 5], [1.0, 2.0, 4.0, 5.0], 2),
            ([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0], [2, 3, 4, 5], [5.0, 7.0, 8.0, 9.0], 0),
        ],
    )
    def test_detect_overlap(self, state_tokens, state_timesteps, new_tokens, new_timesteps, expected_overlap):
        overlap = detect_overlap(state_tokens, state_timesteps, new_tokens, new_timesteps)
        assert overlap == expected_overlap

    @pytest.mark.unit
    def test_merge_word_tail_without_pnc(self):
        word_head = Word(text="meaning", start=0.0, end=1.0, conf=0.5)
        word_tail = Word(text="ful", start=1.0, end=2.0, conf=0.6)
        head, _ = merge_word_tail(word_head, word_tail, conf_aggregator=min)

        assert head.text == "meaningful"
        assert head.start == 0.0
        assert head.end == 2.0
        assert head.conf == 0.5

    @pytest.mark.unit
    def test_merge_word_tail_with_pnc(self):

        word_head = Word(text="meaning", start=0.0, end=1.0, conf=0.5)
        word_tail = Word(text="s", start=1.0, end=2.0, conf=0.6)
        pnc_head = Word(text="Meaning?", start=0.0, end=1.0, conf=0.5)
        new_head, new_pnc_head = merge_word_tail(word_head, word_tail, conf_aggregator=min, pnc_word_head=pnc_head)

        assert new_head.text == "meanings"
        assert new_head.start == 0.0
        assert new_head.end == 2.0
        assert new_head.conf == 0.5
        assert new_pnc_head.text == "Meanings?"
        assert new_pnc_head.start == 0.0
        assert new_pnc_head.end == 2.0
        assert new_pnc_head.conf == 0.5
