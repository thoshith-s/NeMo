# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
dataset_meta_info = {
    'riva_hard_digits': {
        'manifest_path': '/Data/evaluation_manifests/hard-digits-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_letters': {
        'manifest_path': '/Data/evaluation_manifests/hard-letters-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_money': {
        'manifest_path': '/Data/evaluation_manifests/hard-money-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_short': {
        'manifest_path': '/Data/evaluation_manifests/hard-short-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'vctk': {
        'manifest_path': '/Data/evaluation_manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths_silence_trimmed.json',
        'audio_dir': '/Data/VCTK-Corpus-0.92',
        'feature_dir': '/Data/VCTK-Corpus-0.92',
    },
    'libritts_seen': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_seen_evalset_from_testclean_v2.json',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    'libritts_test_clean': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_test_clean_withContextAudioPaths.jsonl',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    # We need an4_val_ci just for CI tests
    'an4_val_ci': {
        'manifest_path': '/home/TestData/an4_dataset/an4_val_context_v1.json',
        'audio_dir': '/',
        'feature_dir': None,
    },
    'es_unseen_small': {
        'manifest_path': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests/es/test_unseen_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/CML/cml_tts_dataset_spanish_v0.1/',
        'feature_dir': None,
        'whisper_language': 'es',
        'tokenizer_names': ['spanish_phoneme'],
        'load_cached_codes_if_available': False
    },
    'fr_unseen_small': {
        'manifest_path': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests/fr/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/CML/cml_tts_dataset_french_v0.1/',
        'feature_dir': None,
        'whisper_language': 'fr',
        'tokenizer_names': ['french_chartokenizer'],
        'load_cached_codes_if_available': False
    },
    'zh_seen_small': {
        'manifest_path': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests/zh/test_rivespeakers_zh_100subset.json',
        'audio_dir': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/riva/zh/',
        'feature_dir': None,
        'whisper_language': 'zh',
        'tokenizer_names': ['mandarin_phoneme'],
        'load_cached_codes_if_available': False
    },
    'vi_seen_small': {
        'manifest_path': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests/vi/manifest_long.json',
        'audio_dir': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/VietnameseEvaluationData/vi',
        'feature_dir': None,
        'whisper_language': 'vi',
        'tokenizer_names': ['vietnamese_phoneme'],
        'load_cached_codes_if_available': False
    },
    'hi_unseen_small': {
        'manifest_path': '/lustre/fsw/llmservice_nemo_speechlm/data/TTS/evaluation_manifests/hi/hindi_100_test.json',
        'audio_dir': '/',
        'feature_dir': None,
        'whisper_language': 'hi',
        'tokenizer_names': ['hindi_phoneme'],
        'load_cached_codes_if_available': False
    },
}
