import requests
import os
from pydub import AudioSegment


def download_audio(url: str, tmp_dir: str, allow_overwrite=False) -> str:

    response = requests.get(url, stream=True)
    response.raise_for_status()

    filename = os.path.basename(url)
    file_path = os.path.join(tmp_dir, filename)

    try:
        # xb is exclusive file creation
        file_write_mode = "xb"
        if allow_overwrite:
            file_write_mode = "wb"
        with open(file_path, file_write_mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except FileExistsError:  # thrown if xb and file exists
        print(f"File {file_path} already exists. Skipping download.")

    return file_path


def do_download(data, directory, operation):
    downloads = []
    for item in data.get("message", {}).get("items", []):
        item_key = item["item_key"]['name']
        for entry in item.get("entries", []):
            audio_path = download_audio(
                entry.get(operation, None).get("source"), directory)
            audio_separation_ref = entry.get(
                operation, None).get("source_separation_ref")
            audio_separation_ref_path = None
            if audio_separation_ref != None:
                audio_separation_ref_path = download_audio(
                    audio_separation_ref, directory)
            language_candidates = entry[operation].get(
                "language_candidates", None)
            learner_duration = entry[operation].get(
                "learner_duration", 0)
            teacher_duration = entry[operation].get(
                "teacher_duration", 0)
            text_assist = entry[operation].get(
                "text_assist", None)
            transcription_language_override_threshold = entry[operation].get(
                "transcription_language_override_threshold", 0)
            transcription_language_reason = entry[operation].get(
                "transcription_language_reason", None)
            language = entry.get(
                operation, None).get("language")
            downloads.append({
                "item_key": item_key,
                "entry_key": entry["key"],
                "audio_path": audio_path,
                "audio_separation_ref_path": audio_separation_ref_path,
                "language_candidates": [language_candidates],
                "learner_duration": learner_duration,
                "teacher_duration": teacher_duration,
                "language": language,
                "text_assist": text_assist,
                "transcription_language_reason": transcription_language_reason,
                "transcription_language_override_threshold": transcription_language_override_threshold
            })
    return downloads


def m4a_to_mp3(m4a_file):
    # Load the M4A file into an AudioSegment object
    audio = AudioSegment.from_file(m4a_file, format="m4a")
    m4a_file_basename = os.path.splitext(os.path.basename(m4a_file))[0]
    dir = os.path.dirname(m4a_file)
    mp3_file = os.path.join(dir, f"{m4a_file_basename}.mp3")
    audio.export(mp3_file, format="mp3")
    return mp3_file
