import os
import csv
import numpy as np
import torch
import torchaudio
from transformers import pipeline
import soundfile as sf
from collections import Counter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

torch.set_default_device("cpu")

MODEL_PATH = "/apps/files/models/Vaani-LID_v0"
LANGID_MATRIX_PATH = "/apps/files/models/langidmatrix.csv"

pipe = pipeline(
    "audio-classification",
    model=MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    device=-1,
)


def load_langid_matrix(filename):
    langid_matrix = {}
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = int(row["en"])
            not_en = int(row["not_en"])
            decision = row["decision"]
            score = int(row["score"])
            confidence = row["confidence"]
            langid_matrix[(en, not_en)] = (decision, score, confidence)

    return langid_matrix


langid_matrix = load_langid_matrix(LANGID_MATRIX_PATH)


def load_audio_files(csv_file):

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # remove this if there is no header

        audio_paths = [row[0] for row in reader]

    return audio_paths


def split_audio(audio_file):

    audio, sample_rate = sf.read(audio_file)
    n = len(audio)

    if n < 6 * sample_rate:
        return [], sample_rate

    if n <= 12 * sample_rate:
        return [audio], sample_rate

    chunk_size = 8 * sample_rate
    min_chunk_size = 5 * sample_rate

    chunks = [
        audio[i:i + chunk_size]
        for i in range(0, n, chunk_size)
    ]

    return [chunk for chunk in chunks if len(chunk) >= min_chunk_size], sample_rate


def run_lang_id(audio_chunks, sample_rate):

    lang_id_raw_output = {}

    for index, chunk in enumerate(audio_chunks):

        chunk_id = index+1  # just so the ids start from 1
        chunk_duration = round(len(chunk)/sample_rate, 1)

        result = pipe(
            {
                "raw": chunk,
                "sampling_rate": sample_rate,
            }
        )

        top_3 = result[:3]
        chunk_lang_info = {
            'id': index+1,
            'duration': chunk_duration
        }
        for item_index, item in enumerate(top_3):
            # so the key is "lang1", "lang2" etc.
            key = "lang" + str(item_index+1)
            score = round(item['score'], 2)
            lang = item['label']
            chunk_lang_info[key] = (lang, score)

        lang_id_raw_output[chunk_id] = chunk_lang_info

    return lang_id_raw_output


def process_audio(audio_file):

    audio_chunks, sample_rate = split_audio(audio_file)

    lang_id_raw_output = run_lang_id(audio_chunks, sample_rate)

    print(lang_id_raw_output)
    return lang_id_raw_output

# remark codes
# 301 = English
# 302 = English, possibly accented articulation, or mixed with some home language vocabulary
# 303 = English, likely heavy home language accent and/or highly indistinct articulation
# 101 = Not English (no additional remark for now)
# 102 = Not English, likely heavy accent and/or indistinct articulation


def language_decision_step_1(lang_id_raw_output):

    lang_decision_step_1 = []

    for key, chunk_lang_info in lang_id_raw_output.items():

        # init outputs for each chunk to default non-english.
        langid = "not_en"
        remark_code = 101

        chunk_id, chunk_duration, lang1, lang2, lang3 = [
            chunk_lang_info[k] for k in ("id", "duration", "lang1", "lang2", "lang3")
        ]

        if (lang1[0] == "English" and lang1[1] > 0.6):
            langid = "en"
            if lang1[1] > 0.85:
                remark_code = 301
            else:
                remark_code = 302

            lang_decision_step_1.append(
                (chunk_id, chunk_duration, langid, lang1[1], remark_code))
            continue

        if (lang1[1] < 0.6 and lang1[1] + lang2[1] + lang3[1] <= 0.8):
            langid = "en"
            remark_code = 303
            lang_decision_step_1.append(
                (chunk_id, chunk_duration, langid, lang1[1], remark_code))
            continue

        # if we reach here, langid is "not_en"
        if (lang1[1]) < 0.6:  # clearly articulated home lang is usually > 0.8
            remark_code = 102

        lang_decision_step_1.append(
            (chunk_id, chunk_duration, langid, lang1[1], remark_code))

    return lang_decision_step_1


def language_decision_step_2(lang_decision_step_1, langid_matrix):

    counts = Counter(item[2] for item in lang_decision_step_1)
    count_en = counts["en"]
    count_not_en = counts["not_en"]
    decision, score, confidence = langid_matrix[(count_en, count_not_en)]
    remarks = str(dict(Counter(item[-1] for item in lang_decision_step_1)))
    return decision, score, confidence, remarks


def run_language_detection(audio_file, learner_duration):

    # default: English not spoken
    lang_detection_output = {
        "langid_decision_data": "",
        "decision": "not_en",
        "score": 1,
        "remark": "000",
        "confidence": "H"
    }

    # edge case - no learner speech separated. no language to detect.
    if (learner_duration == 0):

        # '-' character is used with this special meaning throughout ELA for lang code.
        lang_detection_output['decision'] = '-'
        lang_detection_output['confidence'] = "H"

        return {"lang_detection_output": lang_detection_output}

    lang_id_raw_output = process_audio(audio_file)

    lang_decision_step_1 = language_decision_step_1(lang_id_raw_output)
    decision, score, confidence, remarks = language_decision_step_2(
        lang_decision_step_1, langid_matrix)
    lang_detection_output = {
        "langid_decision_data": lang_decision_step_1,
        "decision": decision,
        "score": score,
        "remark": remarks,
        "confidence": confidence
    }

    # return {"languages_estimation": languages_estimation}
    return {"language_identification": lang_detection_output}
