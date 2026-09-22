# Publish to GitHub

The easiest Windows/Git-Bash path is:

```bash
cd ~/Downloads/llm-gogol-effect-github-ready
bash publish_github.sh
```

The script is rerunnable. It:

1. checks `git`, `gh`, and GitHub authentication;
2. creates `victorlavrenko/llm-gogol-effect` as a **public** repository if absent;
3. clones the current remote into a temporary directory;
4. mirrors the package contents into that clone;
5. commits only if something changed;
6. pushes to `main` without force-pushing.

If `gh` is missing in Windows:

```bash
winget.exe install --id GitHub.cli --exact
```

Authenticate once:

```bash
gh auth login -h github.com -p https -w
```

Then rerun:

```bash
bash publish_github.sh
```

To target a different repository without editing the script:

```bash
REPO=victorlavrenko/another-name bash publish_github.sh
```
