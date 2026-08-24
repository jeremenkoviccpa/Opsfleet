#!/usr/bin/env bash
# Creates a Google Cloud project for the agent and points the repo at it.
#
# Prerequisite (interactive, must be run by you):
#   gcloud auth login
#
# Then:
#   ./scripts/setup_gcp.sh [project-id]
#
# Costs: creating a project is free. BigQuery gives 1 TB of query compute per
# month free; this agent's queries scan megabytes. No billing account is needed
# to query public datasets - the project runs in BigQuery sandbox mode.
set -euo pipefail

PROJECT_ID="${1:-retail-insight-agent-$(date +%s | tail -c 6)}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not on PATH. Install it, or open a new shell if you just did." >&2
  exit 1
fi

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "(unset)" ]; then
  echo "Not signed in. Run this first, then re-run this script:" >&2
  echo "    gcloud auth login" >&2
  exit 1
fi
echo "signed in as: $ACCOUNT"

if gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
  echo "project $PROJECT_ID already exists — reusing it"
else
  echo "creating project $PROJECT_ID ..."
  gcloud projects create "$PROJECT_ID" --name="Retail Insight Agent"
fi

gcloud config set project "$PROJECT_ID" >/dev/null
echo "enabling the BigQuery API (may take ~30s) ..."
gcloud services enable bigquery.googleapis.com --project="$PROJECT_ID" || {
  echo "Could not enable the BigQuery API automatically. Enable it here:" >&2
  echo "    https://console.cloud.google.com/apis/library/bigquery.googleapis.com?project=$PROJECT_ID" >&2
}

echo
echo "Project ready: $PROJECT_ID"
echo
echo "One interactive step remains — the Python client needs its own credentials:"
echo "    gcloud auth application-default login"
echo
echo "Then verify the adapter end to end:"
echo "    GOOGLE_CLOUD_PROJECT=$PROJECT_ID PYTHONPATH=src .venv/bin/python scripts/verify_bigquery.py"
echo
echo "And run the agent against the real dataset:"
echo "    GOOGLE_CLOUD_PROJECT=$PROJECT_ID RIA_WAREHOUSE=bigquery PYTHONPATH=src .venv/bin/python -m agent"
