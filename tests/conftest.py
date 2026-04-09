"""Ensure required env vars exist before any test module imports `src.config.settings`."""
from __future__ import annotations

import os

# AWSSettings requires these; many test imports pull `settings` transitively.
os.environ.setdefault("AWS_S3_RESUME_BUCKET", "test-resume-bucket")
os.environ.setdefault("AWS_S3_JD_BUCKET", "test-jd-bucket")
