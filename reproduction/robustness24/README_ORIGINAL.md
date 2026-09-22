# AI-slop robustness — independent 24-prompt extension

Prompt-bank SHA-256:

`00f76758a881710038e5a80188668fc8df9fc02737482d6c1fe843189b0aa22a`

## Run in Git Bash

```bash
cd ~/Downloads
rm -rf ai_slop_robustness_extension_v1 ai_slop_robustness_extension_run
unzip ai_slop_robustness_extension_v1.zip
cd ~/Downloads/ai_slop_robustness_extension_v1

python ai_slop_robustness_extension_v1.py --self-test

python ai_slop_robustness_extension_v1.py \
  --out ~/Downloads/ai_slop_robustness_extension_run \
  --dry-run
```

Then run the extension:

```bash
python ai_slop_robustness_extension_v1.py \
  --out ~/Downloads/ai_slop_robustness_extension_run
```

The run is resumable. Re-run the same command after provider/API interruptions.

## Package the extension result

```bash
cd ~/Downloads
tar -czf ai_slop_robustness_extension_run.tgz ai_slop_robustness_extension_run
```

## Optional pooled 16 + 24 analysis

After extracting the previous `ai_slop_robustness_run` and completing the extension:

```bash
cd ~/Downloads/ai_slop_robustness_extension_v1

python pool_initial16_extension24.py \
  --initial-run ~/Downloads/ai_slop_robustness_run \
  --extension-run ~/Downloads/ai_slop_robustness_extension_run \
  --out ~/Downloads/ai_slop_robustness_pooled40
```

This reports the initial 16, fresh extension 24, and pooled 40 separately.
