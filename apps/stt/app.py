import sys
import requests
import tempfile
import os
import json
from ela import client as ela_client
from ela import util as ela_util
from stt import transcribe

OPERATION = "stt"


def main(activity_id):

    api_client = ela_client.Client()
    response = api_client.get_data(activity_id, OPERATION)
    data = response.json()

    # important to maintain live context using 'with' for temp directories/files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # download all audio files first, then transcribe in a loop
        downloads = ela_util.do_download(tmp_dir)

        outputs = []

        # after all audio files downloaded, run diarize for each
        for d in downloads:
            asr_text, language, confidence = diarize(
                d["audio_path"], d["language"])
            outputs.append({
                "item_key": d["item_key"],
                "entry_key": d["entry_key"],
                "asr_text": asr_text,
                "language": language,
                "confidence": confidence
            })

        payload = {
            "outputs": outputs
        }
        print(payload)
        api_client.put_data(payload)


if __name__ == "__main__":
    activity_id = sys.argv[1]
    main(activity_id)
