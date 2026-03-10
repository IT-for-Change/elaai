import string
import os
from collections import Counter
import whisper

model = whisper.load_model(
    "large-v3-turbo", download_root="/apps/files/models/whisper")

hallu_thresholds_definition = {
    1: 5,  # Unigrams - occuring 5 or more times
    2: 5,  # Bigrams - occuring 5 or more times
    3: 3,  # Trigrams - occuring 3 or more times
    4: 3,  # Phrases with 4 words - occuring 3 or more times
    5: 2,  # Phrases with 5 or more words - occurring 2 or more times
    6: 2,  # Phrases with 6 or more words - occurring 2 or more times
    7: 2,  # Phrases with 7 or more words - occurring 2 or more times
    8: 2,  # Phrases with 8 or more words - occurring 2 or more times
    9: 2,  # Phrases with 9 or more words - occurring 2 or more times
    10: 2,  # Phrases with 10 or more words - occurring 2 or more times
}


def hallucination_metrics(text):
    hallu_score, hallu_tokens = detect_hallucination(text)
    return hallu_score, hallu_tokens


def get_n_gram_sets(tokens):
    max_n_gram_sets_to_make = len(hallu_thresholds_definition)
    # Step 2: Generate n-grams for each n from 1 to max_n_gram
    ngrams = {}
    for n in range(1, max_n_gram_sets_to_make + 1):
        ngram_list = []
        i = 0
        while i + n <= len(tokens):  # Ensure we don't go out of bounds
            # Join words in the current window to form the n-gram
            ngram = ' '.join(tokens[i:i + n])
            ngram_list.append(ngram)
            i += n  # Move the index forward by N to ensure non-overlapping N-grams
        ngrams[n] = ngram_list

    return ngrams


def check_n_gram_repetition(n, n_gram_set):
    n_gram_token_count = len(n_gram_set)
    threshold_for_n = hallu_thresholds_definition[n]
    hallucination_tokens = []
    is_hallu = False

    # Loop through the array to find consecutive occurrences
    count = 1  # counter to count consecutive tokens
    for i in range(1, n_gram_token_count):
        if n_gram_set[i] == n_gram_set[i - 1]:  # Check if current token equals previous
            count += 1
            hallucination_tokens.append(n_gram_set[i])
        else:
            count = 1  # Reset count when the sequence breaks

        if count >= threshold_for_n:  # If we hit threshold number of consecutive matches
            is_hallu = True

    return is_hallu, hallucination_tokens


def tokens_from_text(text):
    translator = str.maketrans('', '', string.punctuation)
    tokens = text.translate(translator).lower().split()
    return tokens


def detect_hallucination(text):

    total_hallu_tokens = 0
    max_n_gram_sets_to_make = len(hallu_thresholds_definition)
    tokens = tokens_from_text(text)

    ngram_sets = get_n_gram_sets(tokens)

    for n in range(1, max_n_gram_sets_to_make + 1):
        n_gram_set = ngram_sets[n]
        is_hallu, hallu_tokens = check_n_gram_repetition(n, n_gram_set)
        if is_hallu:
            total_hallu_tokens = sum(len(s.split()) for s in hallu_tokens)
            # next, apply correction factor to include the first occurrence since hallucination is confirmed
            total_hallu_tokens += n
            # this breaking out at the first n-gram match for hallucination makes the assumption
            # that there are not goingto be different regions of hallucinations in the
            # transcription involving different sets of word tokens. assumption to be revisited.
            break

    hallu_score = (total_hallu_tokens * 100) // len(tokens)
    return hallu_score, hallu_tokens


def guess_hallucination_text(asr_text):
    return ''


def transcribe(audio_file, language):

    transcription_output = {}

    if language == 'en':
        audio = whisper.load_audio(audio_file)
        options = {
            "language": language,
            "task": "transcribe"
        }
        result = whisper.transcribe(model, audio, **options)
        asr_text = result['text']

        hallu_score, hallu_text = hallucination_metrics(asr_text)

        transcription_output = {
            'asr_text': asr_text,
            'hallu_score': hallu_score,
            'hallu_text': hallu_text
        }
    else:
        # if language detection was not possible or was meaningless or if language was high probability Non English
        transcription_output = {
            'asr_text': '',
            'hallu_score': 0,
            'hallu_text': ''
        }

    return {"transcription_output": transcription_output}
