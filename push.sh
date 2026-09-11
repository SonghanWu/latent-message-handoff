#!/usr/bin/env bash
# Create the GitHub repo and push.  Requires `gh auth login` to have been run once.
#
# Note: the repo is already initialised and committed.  Claude's file bridge cannot
# delete files, so it left behind empty git lock files and could not rename the branch
# from master to main -- the first block below cleans that up.
set -euo pipefail
cd "$(dirname "$0")"

rm -f .git/*.lock .git/refs/heads/*.lock .git/objects/*.lock
rm -rf .git/_claude_stale
find .git/objects -name 'tmp_obj_*' -delete 2>/dev/null || true

git add -A
git diff --cached --quiet || git commit -q --amend --no-edit

git branch -M main
gh repo create latent-message-handoff --public --source=. --remote=origin --push
gh repo view --web
