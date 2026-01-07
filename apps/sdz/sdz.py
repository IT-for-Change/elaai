import os

from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics

from pydub import AudioSegment
import torch
import librosa
from librosa.sequence import dtw
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from ela import util

# disable metrics for current session
set_telemetry_metrics(False)
pipeline = Pipeline.from_pretrained(
    "/apps/sdz/models/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")


def normalize_mfcc(mfcc_features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(mfcc_features)
    return normalized_features


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = normalize_mfcc(mfccs)
    feature_mean = np.mean(mfccs, axis=1)
    return feature_mean


def compute_speaker_distance(speaker_features, teacher_speaker_features):
    distance = euclidean(teacher_speaker_features, speaker_features)
    return distance


def compute_speaker_distance_dtw(speaker_features, teacher_speaker_features):
    distance, _, _, _ = dtw(speaker_features.T, teacher_speaker_features.T)
    return distance


def diarize(audio_file, num_speakers=2):
    audio_file_mp3 = audio_file
    if (audio_file.endswith('.m4a')):
        audio_file_mp3 = util.m4a_to_mp3(audio_file)

    # specifying num_speakers=2
    diarization = pipeline(audio_file_mp3, num_speakers)

    audio = AudioSegment.from_file(audio_file_mp3)
    # empty audio segments for each speaker
    speaker_00 = AudioSegment.empty()
    speaker_01 = AudioSegment.empty()

    # min duration in milliseconds
    min_duration_ms = 500
    speaker_00_duration_ms_total = 0
    speaker_01_duration_ms_total = 0
    speaker_00_max_duration = 0
    speaker_01_max_duration = 0
    total_turns = 0
    speakers = set()
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        total_turns += 1
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        duration_ms = end_ms - start_ms
        speakers.add(speaker)
        if duration_ms >= min_duration_ms:  # Only process segments with duration > min duration
            if speaker == 'SPEAKER_00':
                speaker_00_max_duration = max(
                    duration_ms, speaker_00_max_duration)
                speaker_00_duration_ms_total += duration_ms
                speaker_00 += audio[start_ms:end_ms]
            elif speaker == 'SPEAKER_01':
                speaker_01_max_duration = max(
                    duration_ms, speaker_01_max_duration)
                speaker_01 += audio[start_ms:end_ms]
                speaker_01_duration_ms_total += duration_ms

    audio_path_speaker_0 = audio_file_mp3.replace('.mp3', '_speaker_0.mp3')
    audio_path_speaker_1 = audio_file_mp3.replace('.mp3', '_speaker_1.mp3')

    speaker_00.export(audio_path_speaker_0, format='mp3')
    speaker_01.export(audio_path_speaker_1, format='mp3')

    diarization_output = {
        'speaker_00_audio_path': audio_path_speaker_0,
        'speaker_01_audio_path': audio_path_speaker_1,
        'speaker_00_duration': speaker_00_duration_ms_total,
        'speaker_01_duration': speaker_01_duration_ms_total,
        'speaker_00_max_continuous_duration': speaker_00_max_duration,
        'speaker_01_max_continuous_duration': speaker_01_max_duration,
        'total_turns': total_turns,
        'speakers': speakers
    }
    return diarization_output


def identify_speakers(diarization_output, teacher_audio_sample_file):

    audio_path_speaker_0 = diarization_output['speaker_00_audio_path']
    audio_path_speaker_1 = diarization_output['speaker_01_audio_path']

    speakers_info = {}
    teacher_speaker_features = extract_features(teacher_audio_sample_file)
    speaker_0_features = extract_features(audio_path_speaker_0)
    speaker_1_features = extract_features(audio_path_speaker_1)
    distance_speaker_0 = compute_speaker_distance(
        speaker_0_features, teacher_speaker_features)
    distance_speaker_1 = compute_speaker_distance(
        speaker_1_features, teacher_speaker_features)
    if distance_speaker_0 < distance_speaker_1:
        speakers_info['audio_filepath_learner'] = audio_path_speaker_1
        speakers_info['audio_filepath_teacher'] = audio_path_speaker_0
        speakers_info['learner_duration'] = diarization_output['speaker_01_duration']
        speakers_info['learner_max_duration'] = diarization_output['speaker_01_max_continuous_duration']
        speakers_info['teacher_duration'] = diarization_output['speaker_00_duration']
        speakers_info['teacher_max_duration'] = diarization_output['speaker_00_max_continuous_duration']
    else:
        speakers_info['audio_filepath_learner'] = audio_path_speaker_0
        speakers_info['audio_filepath_teacher'] = audio_path_speaker_1
        speakers_info['learner_duration'] = diarization_output['speaker_00_duration']
        speakers_info['learner_max_duration'] = diarization_output['speaker_00_max_continuous_duration']
        speakers_info['teacher_duration'] = diarization_output['speaker_01_duration']
        speakers_info['teacher_max_duration'] = diarization_output['speaker_01_max_continuous_duration']

    speakers_info['total_turns'] = diarization_output['total_turns']
    return speakers_info
