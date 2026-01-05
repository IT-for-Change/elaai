from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
import time

print('disablig telemetry', flush=True)
# disable metrics for current session
set_telemetry_metrics(False)
print('loading model', flush=True)
pipeline = Pipeline.from_pretrained(
    "/apps/sdz/models/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")
print('model loaded', flush=True)
print('sleeping..', flush=True)
time.sleep(240)
