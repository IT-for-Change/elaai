import numpy as np
from ast import literal_eval


def extract_en_audio(audio, langid_chunk_data, logit_threshold=0.6):

    sample_rate = 16000
    selected_audio = []

    current_time = 0.0

    langid_chunk_data = literal_eval(langid_chunk_data)

    for chunk_number, duration, language, logit, *_ in langid_chunk_data:

        start_time = current_time
        end_time = current_time + duration

        if language == "en" and logit >= logit_threshold:

            start_sample = round(start_time * sample_rate)
            end_sample = round(end_time * sample_rate)

            chunk_audio = audio[start_sample:end_sample]

            selected_audio.append(chunk_audio)

        current_time = end_time

    if not selected_audio:
        return None

    return np.concatenate(selected_audio).astype(np.float32)
