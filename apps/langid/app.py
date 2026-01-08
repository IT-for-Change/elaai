import sys
import requests
import tempfile
import os
import json
from loguru import logger
from ela import client as ela_client
from ela import util as ela_util
# from sdz.sdz import diarize, identify_speakers

OPERATION = "langid"


def main(activity_id):

    logger.info('Initializing client')
    api_client = ela_client.Client()

    logger.info('Client get_data')
    response = api_client.get_data(activity_id, OPERATION)
    data = response.json()
    logger.info('Received response')

    # important to maintain live context using 'with' for temp directories/files
    with tempfile.TemporaryDirectory() as tmp_dir:

        # download all audio files first, then transcribe in a loop
        downloads = ela_util.do_download(data, tmp_dir, OPERATION)

        outputs = []
        # after all audio files downloaded, run speaker diarization for each
        for d in downloads:
            logger.info(
                f'Performing language check for submission {d["item_key"]} and entry {d["entry_key"]}')
            # diarization_output = diarize(d["audio_path"])
            logger.info(diarization_output)
            logger.info('Running speaker identification')
            speakers_info = identify_speakers(diarization_output,
                                              d["audio_separation_ref_path"])
            logger.info(speakers_info)

            outputs.append({
                "item_key": d["item_key"],
                "entry_key": d["entry_key"],
                "langid": speakers_info
            })

            if len(outputs) > 1:
                break

        # print(outputs)
        api_client.put_data(outputs, OPERATION)


if __name__ == "__main__":
    activity_id = sys.argv[1]
    main(activity_id)
