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

from itertools import groupby
from typing import Iterable, Union
import logging

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
)
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class HybridSALMTDTDataset(torch.utils.data.Dataset):
    """
    A dataset for Hybrid SALM-TDT models that separates speech and non-speech data.
    
    This dataset processes multimodal conversations and separates them into:
    1. Speech data: goes through both TDT and SALM heads
    2. Non-speech data: goes through SALM head only
    
    Args:
        salm_tokenizer (AutoTokenizer): SALM tokenizer for language model
        tdt_tokenizer (AutoTokenizer): TDT tokenizer for ASR model
    """

    def __init__(self, salm_tokenizer: AutoTokenizer, tdt_tokenizer: AutoTokenizer) -> None:
        self.salm_tokenizer = salm_tokenizer
        self.tdt_tokenizer = tdt_tokenizer
        self.salm_pad_id = get_pad_id(salm_tokenizer)
        self.tdt_pad_id = get_pad_id(tdt_tokenizer)
        self.debug_file_count = 0  # Limit debug file creation

    def __getitem__(self, conversations: CutSet) -> dict | None:
        if not conversations or len(conversations) == 0:
            # Return None for empty conversations - FallbackDataset will handle this
            print(f"WARNING: Empty conversations provided to dataset - returning None for fallback handling")
            return None
        
        # Check batch homogeneity
        tdt_count = sum(1 for c in conversations if hasattr(c, 'tdt_input_ids') and c.tdt_input_ids is not None)
        total_count = len(conversations)
        
        # Ensure batch consistency:
        # Samples with the "transcribe" prompt have their inputs tokenized using the TDT tokenizer and stored in "tdt_input_ids".
        # If any sample in the batch contains "tdt_inputs", then all samples in that batch must also include "tdt_inputs".
        # This avoids training hangs caused when only a subset of GPUs computes the TDT
        if tdt_count != 0 and tdt_count != total_count:
            import os
            import time
            debug_file = f"debug_mixed_batch_pid{os.getpid()}_{int(time.time())}.txt"
            with open(debug_file, "w") as f:
                f.write(f"MIXED BATCH DETECTED: {tdt_count}/{total_count} conversations have TDT data\n")
                f.write("All conversations in batch:\n")
                for i, c in enumerate(conversations):
                    has_tdt = hasattr(c, 'tdt_input_ids') and c.tdt_input_ids is not None
                    conv_id = getattr(c, 'id', 'unknown')
                    f.write(f"  Conv {i}: ID={conv_id}, has_tdt={has_tdt}\n")
                    f.write(f"    Conv type: {type(c)}\n")
                    f.write("-" * 40 + "\n")
            raise ValueError(f"Mixed batch: {tdt_count}/{total_count} conversations have TDT data. Debug info: {debug_file}")
        
        # Process conversations - this may filter out conversations with audio loading issues
        original_count = len(conversations)
        audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(conversations)
        
        # Debug audio loading issues
        if len(conversations) < original_count:
            filtered_count = original_count - len(conversations)
            print(f"WARNING: {filtered_count}/{original_count} conversations filtered out due to audio loading issues")
            
        if len(conversations) == 0:
            print(f"CRITICAL: All {original_count} conversations filtered out due to audio loading failures!")
            print(f"Returning None for FallbackDataset to handle...")
            return None

        salm_input_ids = left_collate_vectors([c.input_ids for c in conversations], padding_value=self.salm_pad_id)        
                
        if tdt_count > 0:
            tdt_input_ids = right_collate_vectors([c.tdt_input_ids for c in conversations], padding_value=0)
            tdt_input_ids_len = torch.tensor([c.tdt_input_ids_len for c in conversations], device=tdt_input_ids.device, dtype=torch.long)
            
            return {
                "audios": audios,
                "audio_lens": audio_lens,
                "input_ids": salm_input_ids,  # For SALM head
                "tdt_input_ids": tdt_input_ids,    # For TDT head
                "tdt_input_ids_len": tdt_input_ids_len,
                "loss_mask": left_collate_vectors(
                    [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
                ).to(torch.bool),
                "conversations": drop_in_memory_data(conversations),
            }
        else:
            return {
                "audios": audios,
                "audio_lens": audio_lens,
                "input_ids": salm_input_ids,  # For SALM head
                "loss_mask": left_collate_vectors(
                    [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
                ).to(torch.bool),
                "conversations": drop_in_memory_data(conversations),
            }

    def _extract_conversation_text(self, conversation) -> str:
        """Extract raw text content from a conversation for TDT tokenization (without prompts)."""
        text_parts = []
        for turn in conversation.turns:
            if hasattr(turn, 'value'):  # TextTurn
                # For TDT, we want the raw text content without any prompt formatting
                text_parts.append(turn.value)
            elif hasattr(turn, 'audio_locator_tag'):  # AudioTurn
                # For audio turns in TDT training, we want the transcription text
                if hasattr(turn, 'text') and turn.text:
                    # Use the transcription text from the AudioTurn
                    text_parts.append(turn.text)
                else:
                    # Fallback to audio_locator_tag if no transcription text available
                    # This might need adjustment based on your specific data format
                    text_parts.append(turn.audio_locator_tag)
        return " ".join(text_parts)


def left_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")


def right_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="right")

def drop_in_memory_data(conversations: CutSet) -> CutSet:
    def _drop(conversation: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
        turns = []
        for t in conversation.turns:
            if isinstance(t, AudioTurn):
                t = fastcopy(t, cut=t.cut.drop_in_memory_data())
            turns.append(t)
        return fastcopy(conversation, turns=turns)

    return conversations.map(_drop, apply_fn=None)