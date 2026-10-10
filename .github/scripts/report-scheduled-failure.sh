#!/usr/bin/env bash
# Turn a failed scheduled run into an open issue.
#
# A scheduled run that fails notifies nobody who is watching: the weekly
# dependency scan stayed red on main for five weeks before anyone looked.
# One issue per workflow; further failures are added as comments rather than
# opening duplicates.
#
# Needs: GH_TOKEN (issues: write), GH_REPO, WORKFLOW_NAME, RUN_URL.
set -euo pipefail

title="Scheduled workflow failing: ${WORKFLOW_NAME}"
body="The scheduled run of **${WORKFLOW_NAME}** failed: ${RUN_URL}"

gh label create scheduled-failure \
  --description "A scheduled workflow is failing on main" \
  --color B60205 2>/dev/null || true

# Listings trail writes by a few seconds, so two calls in quick succession can
# miss each other. Scheduled runs are a day or more apart, which is ample.
existing=$(gh api "repos/${GH_REPO}/issues?state=open&labels=scheduled-failure&per_page=100" \
  --jq "map(select(.pull_request == null and .title == \"${title}\")) | .[0].number // empty")

if [ -n "${existing}" ]; then
  gh issue comment "${existing}" --body "Still failing. ${body}"
else
  gh issue create --title "${title}" --label scheduled-failure --body "${body}

This issue was opened automatically. Further failures are added as comments. Close it once a scheduled run passes."
fi
