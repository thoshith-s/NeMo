# Streaming / Buffered / Chunked ASR

Contained within this directory are scripts to perform streaming or buffered inference of audio files using CTC / Transducer ASR models, and chunked inference for MultitaskAED models (e.g., "nvidia/canary-1b").

## Difference between streaming and buffered ASR

While we primarily showcase the defaults of these models in buffering mode, note that the major difference between streaming ASR and buffered ASR is the chunk size and the total context buffer size.

If you reduce your chunk size, the latency for your first prediction is reduced, and the model appears to predict the text with shorter delay. On the other hand, since the amount of information in the chunk is reduced, it causes higher WER.

On the other hand, if you increase your chunk size, then the delay between spoken sentence and the transcription increases (this is buffered ASR). While the latency is increased, you are able to obtain more accurate transcripts since the model has more context to properly transcribe the text.

## Chunked Inference

For MultitaskAED models, we provide a script to perform chunked inference. This script will split the input audio into non-overlapping chunks and perform inference on each chunk. The script will then concatenate the results to provide the final transcript.

## Universal Streaming Inference

We also provide a script (`asr_streaming_infer.py`) for performing streaming inference with buffered (CTC/RNNT/TDT) and cache-aware (CTC/RNNT) models. This script can process a single audio file, a directory of audio files, or a manifest file. In addition to streaming ASR, the script supports Punctuation and Capitalization, Inverse Text Normalization, and End of Utterance detection. It also provides both word-level and segment-level output. All related configurations are available in the `../conf/asr_chunked_inference/` directory.
