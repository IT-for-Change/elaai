import os
import sys

from pyannote.audio import Model, Inference
from pyannote.core import Segment
from pyannote.audio.telemetry import set_telemetry_metrics

from pydub import AudioSegment
import torch
import torchaudio
import librosa
from librosa.sequence import dtw
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks, save_audio

# disable metrics for current session
set_telemetry_metrics(False)

model = Model.from_pretrained(
    "/apps/files/models/pyannote/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/embedding")
inference_model = Inference(model, window="whole", device=torch.device("cpu"))

TARGET_SAMPLE_RATE = 16000


def write_waveform_16K(audio_file, stereo=False):

    audio_file_wave = audio_file.replace('.m4a', '.wav')
    waveform = resample(audio_file)
    torchaudio.save(audio_file_wave, waveform, TARGET_SAMPLE_RATE)
    return audio_file_wave


def resample(audio_file, stereo=False):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE)
    # mono
    if (stereo):
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def teacher_audio_reference_waveform(teacher_sample_audio):

    ref_waveform = resample(teacher_sample_audio)

    ref_input = {
        "waveform": ref_waveform,  # shape: (channels, samples)
        "sample_rate": TARGET_SAMPLE_RATE
    }

    # Compute embedding for speaker 1
    ref_embedding = inference_model(ref_input)
    ref_embedding = torch.tensor(ref_embedding)
    ref_embedding = ref_embedding / ref_embedding.norm()  # normalize
    return ref_embedding


def run_vad(audio_file):

    model = load_silero_vad()
    waveform = resample(audio_file)
    waveform_file = write_waveform_16K(audio_file)
    waveform = read_audio(waveform_file, TARGET_SAMPLE_RATE)
    speech_timestamps = get_speech_timestamps(
        waveform, model, threshold=0.4, return_seconds=True)  # 0.5 is default. reduce to add a little silence either side to reduce cutting out of words

    return waveform, speech_timestamps


def separate_speakers(audio_file, teacher_audio_sample_file):

    waveform, speech_timestamps = run_vad(audio_file)
    print(speech_timestamps)

    # Create chunks based on speech segments with small overlap
    speech_chunks = []

    # Process each speech segment
    for i in range(len(speech_timestamps)):
        start = speech_timestamps[i]['start']
        end = speech_timestamps[i]['end']
        print(type(start), start)

        # Convert to sample indices
        start_idx = int(start * TARGET_SAMPLE_RATE)
        end_idx = int(end * TARGET_SAMPLE_RATE)

        # Chunk to be extracted with sample rate not seconds
        conv_chunk = waveform[start_idx:end_idx]
        # But append the start and end in seconds so durations can be computed easily next.
        speech_chunks.append((start, end, conv_chunk))

    teacher_embedding = teacher_audio_reference_waveform(
        teacher_audio_sample_file)

    teacher_speech_segments = []
    learner_speech_segments = []

    learner_duration_total = 0
    teacher_duration_total = 0
    learner_max_duration = 0
    teacher_max_duration = 0
    total_turns = 0
    speakers_info = {}

    for start, end, conversation_chunk in speech_chunks:
        print(
            f"Chunk: {start}s - {end}s")
        turn_duration = end - start
        total_turns += 1

        # Convert the chunk into a 2D tensor with shape (1, time). '1' for mono.
        conversation_chunk_tensor = torch.tensor(
            conversation_chunk).unsqueeze(0)  # Add a dimension for channels
        # Ensure it's a float tensor (if it's not already)
        conversation_chunk_tensor = conversation_chunk_tensor.float()

        chunk_input = {"waveform": conversation_chunk_tensor,
                       "sample_rate": TARGET_SAMPLE_RATE}
        conversation_chunk_embedding = inference_model(chunk_input)
        conversation_chunk_embedding = torch.tensor(
            conversation_chunk_embedding)
        conversation_chunk_embedding = conversation_chunk_embedding / \
            conversation_chunk_embedding.norm()

        # Cosine similarity with speaker 1 reference
        cos_sim = torch.nn.functional.cosine_similarity(
            conversation_chunk_embedding, teacher_embedding, dim=0)
        print(f'{cos_sim}')

        if cos_sim.item() > 0.25:  # threshold for similarity
            teacher_speech_segments.append(conversation_chunk_tensor)
            teacher_duration_total += turn_duration
            teacher_max_duration = max(
                turn_duration, teacher_max_duration)
        else:
            learner_speech_segments.append(conversation_chunk_tensor)
            learner_duration_total += turn_duration
            learner_max_duration = max(
                turn_duration, learner_max_duration)

    teacher_speech = torch.cat(
        teacher_speech_segments, dim=1) if teacher_speech_segments else torch.empty(1, 0)
    learner_speech = torch.cat(
        learner_speech_segments, dim=1) if learner_speech_segments else torch.empty(1, 0)

    audio_path_teacher = audio_file.replace('.m4a', '_teacher_vad.wav')
    audio_path_learner = audio_file.replace('.m4a', '_learner_vad.wav')
    torchaudio.save(audio_path_teacher, teacher_speech, TARGET_SAMPLE_RATE)
    torchaudio.save(audio_path_learner, learner_speech, TARGET_SAMPLE_RATE)

    speakers_info['audio_filepath_learner'] = audio_path_learner
    speakers_info['audio_filepath_teacher'] = audio_path_teacher
    speakers_info['learner_duration'] = int(learner_duration_total)
    speakers_info['learner_max_duration'] = int(learner_max_duration)
    speakers_info['teacher_duration'] = int(teacher_duration_total)
    speakers_info['teacher_max_duration'] = int(teacher_max_duration)
    speakers_info['total_turns'] = total_turns

    return speakers_info
