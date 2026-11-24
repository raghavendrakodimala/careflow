"""Test-wide setup. Tracing is off unless a test asks for it, so the suite never writes out/traces.jsonl."""

import os

os.environ["CAREFLOW_TRACING"] = "off"
