#!/usr/bin/env bash
# Rerunnable publisher for Windows Git Bash, macOS, or Linux.
# Creates the public repository if it does not exist, then mirrors this package
# into the repository and pushes a normal commit to main. It never force-pushes.
set -Eeuo pipefail

REPO="${REPO:-victorlavrenko/llm-gogol-effect}"
DESCRIPTION="${DESCRIPTION:-Reproduction package for The Gogol Effect in LLMs: Post-Completion Self-Devaluation}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' is required but was not found." >&2
    return 1
  fi
}

need git || {
  echo "Install Git for Windows: https://git-scm.com/download/win" >&2
  exit 2
}
need gh || {
  echo "Install GitHub CLI, then rerun:" >&2
  echo "  winget.exe install --id GitHub.cli --exact" >&2
  echo "or download it from https://cli.github.com/" >&2
  exit 2
}
need tar || exit 2

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run:" >&2
  echo "  gh auth login -h github.com -p https -w" >&2
  echo "Then rerun ./publish_github.sh" >&2
  exit 3
fi

LOGIN="$(gh api user --jq .login)"
OWNER="${REPO%%/*}"
if [[ "$LOGIN" != "$OWNER" ]]; then
  echo "WARNING: authenticated as '$LOGIN', but target repository owner is '$OWNER'." >&2
  echo "The script will continue only if your account has permission to create/push there." >&2
fi

if gh repo view "$REPO" >/dev/null 2>&1; then
  echo "Repository exists: https://github.com/$REPO"
else
  echo "Creating public repository: $REPO"
  gh repo create "$REPO" --public --description "$DESCRIPTION"
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT
WORK="$TMP/repo"

echo "Cloning $REPO..."
gh repo clone "$REPO" "$WORK" -- --quiet || {
  # Newly-created empty repositories can be cloned by git, but keep a fallback.
  mkdir -p "$WORK"
  git -C "$WORK" init -q
  git -C "$WORK" remote add origin "https://github.com/$REPO.git"
}

# Ensure we commit to main. For a non-empty repository, clone has already
# checked out its default branch; switching to main is safe when it exists.
if git -C "$WORK" show-ref --verify --quiet refs/remotes/origin/main; then
  git -C "$WORK" checkout -q -B main origin/main
else
  git -C "$WORK" checkout -q -B main
fi

# Mirror the package into the temporary clone. This makes reruns idempotent and
# also removes stale repository files that are no longer present in the package.
find "$WORK" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
(
  cd "$ROOT"
  tar --exclude-vcs --exclude='./*.zip' --exclude='./*.tgz' --exclude='./*.tar.gz' -cf - .
) | (
  cd "$WORK"
  tar -xf -
)

# Prevent accidental secret publication even if a user later adds local files.
find "$WORK" -type f \( -name '.env' -o -name '.env.*' \) -print -delete

if ! git -C "$WORK" config user.name >/dev/null; then
  git -C "$WORK" config user.name "Victor Lavrenko"
fi
if ! git -C "$WORK" config user.email >/dev/null; then
  git -C "$WORK" config user.email "victor@peacetech.vc"
fi

git -C "$WORK" add -A
if git -C "$WORK" diff --cached --quiet; then
  echo "No changes to publish. Repository is already up to date."
else
  COMMIT_MSG="${COMMIT_MSG:-Update paper and reproduction package}"
  git -C "$WORK" commit -m "$COMMIT_MSG"
  git -C "$WORK" push -u origin main
fi

echo
echo "Published: https://github.com/$REPO"
echo "Paper:     https://github.com/$REPO/blob/main/paper/paper.pdf"
echo "Data:      https://github.com/$REPO/tree/main/reproduction"
