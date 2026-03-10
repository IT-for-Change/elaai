import sys
import time
import requests
import tempfile
import os
import json
from loguru import logger
from ela import client as ela_client
from ela import util as ela_util
from report.reporting import do_report


OPERATION = "report"


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
            report_inputs = entry[OPERATION]
            logger.info(
                f'Running report for submission {item_key} and entry {entry["key"]}')
            logger.info(f'report_input: {report_inputs}')
            report_data = do_report(report_inputs)
            logger.info(
                f'Completed report for submission {item_key} and entry {entry["key"]}')
            logger.info(report_data)
            outputs.append({
                "item_key": item_key,
                "entry_key": entry["key"],
                "report": report_data
            })

        print(outputs)
    api_client.put_data(outputs, OPERATION)


if __name__ == "__main__":
    activity_id = sys.argv[1]
    main(activity_id)
