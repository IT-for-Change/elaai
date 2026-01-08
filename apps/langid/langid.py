from collections import defaultdict
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
from pydub.silence import detect_silence, split_on_silence
import librosa


processor = WhisperProcessor.from_pretrained(
    "/apps/files/models/whisper/large-v3-turbo")
model = WhisperForConditionalGeneration.from_pretrained(
    "/apps/files/models/whisper/large-v3-turbo")


def lang_detect(audio_segment, sr, processor, model, language_tokens, device):

    input_features = processor(
        audio_segment, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    language_token_ids = processor.tokenizer.convert_tokens_to_ids(
        language_tokens)
    logits = model(input_features,
                   decoder_input_ids=torch.tensor([[50258] for _ in range(input_features.shape[0])],
                                                  device=device)).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')
    output_probs = logits.softmax(dim=-1).cpu()
    lang_probs = {
        lang[2:-2]: output_probs[0, 0, token_id].item()
        for token_id, lang in zip(language_token_ids, language_tokens)
    }

    return lang_probs


def m4a_to_mp3(m4a_file):
    # Load the M4A file into an AudioSegment object
    audio = AudioSegment.from_file(m4a_file, format="m4a")
    m4a_file_basename = os.path.splitext(os.path.basename(m4a_file))[0]
    dir = os.path.dirname(m4a_file)
    mp3_file = os.path.join(dir, f"{m4a_file_basename}_langid.mp3")
    audio.export(mp3_file, format="mp3")
    return mp3_file


def preprocess_audio(audio_file_path):

    audio_file_path = m4a_to_mp3(audio_file_path)
    audio = AudioSegment.from_mp3(audio_file_path)

    silence_thresh = audio.dBFS  # Silence threshold in dB
    min_silence_len = 100  # Minimum silence length (in milliseconds)

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh)

    preprocessed_audio = AudioSegment.empty()
    for chunk in chunks:
        preprocessed_audio += chunk

    preprocessed_audio.export(audio_file_path, format="mp3")

    return audio_file_path

# CHECK THIS 9/apr/2025
# https://discuss.huggingface.co/t/language-detection-with-whisper/26003/14
# https://huggingface.co/openai/whisper-large-v2/discussions/40
# https://community.openai.com/t/whisper-is-there-a-way-to-tell-the-language-before-recognition/70687/4
# TODO


def lang_detect(audio, sr, processor, model, language_tokens, segment_duration=30):

    segment_length = int(segment_duration * sr)
    segments = [audio[i:i + segment_length]
                for i in range(0, len(audio), segment_length)]

    overall_probs = defaultdict(float)
    device = torch.device("cpu")
    model = model.to(device)

    for i, segment in enumerate(segments):
        lang_probs = lang_detect(
            segment, sr, processor, model, language_tokens, device)

        for lang, prob in lang_probs.items():
            overall_probs[lang] += prob

    # Average the probabilities
    for lang in overall_probs:
        overall_probs[lang] /= len(segments)

    return dict(overall_probs)


def detect_languages(audio_path, preprocess=False):

    # add english!
    language_codes.insert(0, 'en')
    language_tokens = [f'<|{code}|>' for code in language_codes]
    audio_path = submission.media
    print(f'Processing {audio_path} with language codes {language_codes}')
    threshold = 0.15

    if (preprocess):
        # preprocessing:
        # convert to mp3 and remove silences to improve accuracy of detection.
        # Silence removal is emperically observed to improve lang id accuracy significantly
        # this may not be required for conversation activities since the pyannote pipeline does a good job
        # of removing trailing silences between speaker turns.

        # audio_path = preprocess_audio(audio_path)
        pass

    audio, sr = librosa.load(audio_path, sr=16000)
    language_probabilities = lang_detect(
        audio, sr, processor, model, language_tokens)

    language_identification = []
    for lang_code, prob in language_probabilities.items():
        lang_id = {}
        lang_id['language_code'] = lang_code
        lang_id['confidence'] = round(prob, 2)
        language_identification.append(lang_id)

    return language_identification
