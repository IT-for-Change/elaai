#!/bin/bash
set -e

case "$APP_ENV" in
  whisper)
    echo "Using Whisper environment"
    exec /opt/venvs/whisper/bin/python -u "$@"
    ;;
  pyannote)
    echo "Using pyannote environment"
    exec /opt/venvs/pyannote/bin/python -u "$@"
    ;;
  spacy)
    echo "Using spaCy environment"
    exec /opt/venvs/spacy/bin/python -u "$@"
    ;;
  *)
    echo "ERROR: APP_ENV must be one of: whisper | pyannote | spacy"
    exit 1
    ;;
esac

