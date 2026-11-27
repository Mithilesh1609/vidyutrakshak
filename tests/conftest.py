"""Run the tests on their own data, models and console payload, so running
pytest never changes the demo state in data/, models/ or static/."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="vr-tests-")
os.environ["VR_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["VR_MODELS_DIR"] = os.path.join(_TMP, "models")
os.environ.pop("VR_API_KEYS", None)
os.environ["VR_CHAT_LLM"] = "0"  # assistant tests use the offline rules unless they stub the API
