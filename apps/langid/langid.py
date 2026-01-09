from collections import defaultdict
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


processor = WhisperProcessor.from_pretrained(
    "/apps/files/models/whisper/large-v3-turbo")
model = WhisperForConditionalGeneration.from_pretrained(
    "/apps/files/models/whisper/large-v3-turbo")

device = torch.device("cpu")
model = model.to(device)


def lang_detect(audio_segment, sr, language_tokens):

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

# CHECK THIS 9/apr/2025
# https://discuss.huggingface.co/t/language-detection-with-whisper/26003/14
# https://huggingface.co/openai/whisper-large-v2/discussions/40
# https://community.openai.com/t/whisper-is-there-a-way-to-tell-the-language-before-recognition/70687/4
# TODO


def lang_detect_in_segments(audio, sr, language_tokens, segment_duration=30):

    if segment_duration == 0:
        lang_probs = lang_detect(audio, sr, language_tokens)
        return lang_probs

    # else, split into segments for sampling
    segment_length = int(segment_duration * sr)
    segments = [audio[i:i + segment_length]
                for i in range(0, len(audio), segment_length)]

    overall_probs = defaultdict(float)

    for i, segment in enumerate(segments):
        lang_probs = lang_detect(
            segment, sr, language_tokens)

        for lang, prob in lang_probs.items():
            overall_probs[lang] += prob

    # Average the probabilities
    for lang in overall_probs:
        overall_probs[lang] /= len(segments)

    return dict(overall_probs)


def detect_languages(audio_path, language_candidates, spreprocess=False):

    # add english!
    language_candidates.insert(0, 'en')
    language_tokens = [f'<|{code}|>' for code in language_candidates]
    print(
        f'Processing {audio_path} with language candidates {language_candidates}')

    audio, sr = librosa.load(audio_path, sr=16000)
    language_probabilities = lang_detect_in_segments(
        audio, sr, language_tokens, segment_duration=0)

    languages_estimation = []
    for lang_code, prob in language_probabilities.items():
        lang_id = {}
        lang_id['language_code'] = lang_code
        lang_id['confidence'] = round(prob, 2)
        languages_estimation.append(lang_id)

    return {"languages_estimation": languages_estimation}
