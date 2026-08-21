
#!/usr/bin/env bash
# Launches a standalone JupyterLab server for this project.
# Can be run from anywhere -- it cd's into its own directory first.
set -euo pipefail

cd "$(dirname "$0")"
jupyter lab
