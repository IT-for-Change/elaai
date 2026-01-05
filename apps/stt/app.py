import sys
import requests
import tempfile
import os
import json
from stt import transcribe

# ------------------------------------------------------------------
# Stub for your processing function
# ------------------------------------------------------------------


def execute(audio_file_path: str) -> str:
    # Replace with real logic
    return f"processed:{os.path.basename(audio_file_path)}"


# ------------------------------------------------------------------
# Download audio file to a temporary directory
# ------------------------------------------------------------------
def download_audio(url: str, tmp_dir: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    filename = os.path.basename(url)
    file_path = os.path.join(tmp_dir, filename)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return file_path


def main(host, port, api, activity):

    remote_url = f'http://{host}:{port}/api/method/{api}'
    #"http://elaexplore.localhost:8081/api/method/ela.ela.doctype.activity.activity.get_submissions"
    headers = {"Authorization": "token b909eccbd9b04d1:e7515a47bbfad81"}
    params = {"activity_eid": activity, "reason": "stt"}

    response = requests.get(remote_url, params, headers=headers)

    response.raise_for_status()
    data = response.json()

    # v. important to maintain live context using 'with' for temp directories/files
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloads = []
        # download all audio files first, then transcribe in a loop
        for item in data.get("message", {}).get("items", []):
            item_key = item["item_key"]['name']
            for entry in item.get("entries", []):
                audio_path = download_audio(entry["source"], tmp_dir)

                downloads.append({
                    "item_key": item_key,
                    "entry_key": entry["entry_key"],
                    "audio_path": audio_path,
                    "language": entry["language"]
                })

        outputs = []

        # after all audio files downloaded, run execute for each
        for d in downloads:
            asr_text, language, confidence = transcribe(
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

        update_response = requests.post(
            "http://elaexplore.localhost:8081/api/method/ela.ela.doctype.activity.activity.update_submissions",
            json=payload,
            headers=headers
        )
        # update_response.raise_for_status()

        print({update_response.text})


if __name__ == "__main__":
    host = sys.argv[1]
    port = sys.argv[2]
    api = sys.argv[3]
    activity = sys.argv[4]
    main(host, port, api, activity)
