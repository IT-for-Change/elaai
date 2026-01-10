import sys
import time
import requests
import tempfile
import os
import json
from loguru import logger
from ela import client as ela_client
from ela import util as ela_util
from nlp.nlp import analyze_text


OPERATION = "nlp"


def main(activity_id):

    logger.info('Initializing client')
    api_client = ela_client.Client()
    logger.info('Client get_data')
    response = api_client.get_data(activity_id, OPERATION)
    data = response.json()
    logger.info('Received response')

    # time.sleep(300)

    outputs = []
    request_items = data.get("message", {}).get("items", [])
    for item in request_items:
        item_key = item["item_key"]['name']
        entries = item.get("entries", [])
        for entry in entries:
            asr_text = entry[OPERATION]['source']
            language_code = entry[OPERATION]['language']
            check_grammar = entry[OPERATION]['grammar']
            logger.info(
                f'Performing text analysis for submission {item_key} and entry {entry["key"]} in language {language_code}')
            logger.info(f'ASR TEXT: {asr_text}')
            analyzed_text = analyze_text(
                asr_text, language_code, check_grammar)
            logger.info(
                f'Completed text analysis for submission {item_key} and entry {entry["key"]}')
            logger.info(analyzed_text)
            outputs.append({
                "item_key": item_key,
                "entry_key": entry["key"],
                "nlp": analyzed_text
            })

        if len(outputs) > 1:
            logger.info('Reached outputs > 1')
            break

        # print(outputs)
    api_client.put_data(outputs, OPERATION)


if __name__ == "__main__":
    activity_id = sys.argv[1]
    main(activity_id)
