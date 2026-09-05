#!/usr/bin/env bash
# =============================================================================
# Fetch model weights.
#
#   bash scripts/fetch_weights.sh --dry-run     # print targets, no network
#   bash scripts/fetch_weights.sh               # download
#   bash scripts/fetch_weights.sh --base-url https://your.mirror/indra
#
# This is the command MissingCheckpointError names.
#
# Weights are hosted outside the repository -- git is the wrong place for
# several gigabytes of binary that change on every retrain. Override the
# source with --base-url or INDRA_WEIGHTS_BASE_URL to pull from a mirror, an
# internal bucket or a local path.
#
# Reachability is checked before anything is written. Without that, curl saves
# an HTML error page as indra_fusion_v1.pt and the failure surfaces much later
# as an unpickling error that says nothing about the download.
#
# Nothing here runs automatically. No install step, no container build and no
# import invokes this: fetching several gigabytes is a decision an operator
# makes deliberately.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${INDRA_WEIGHTS_BASE_URL:-https://huggingface.co/indra-nowcasting/earthformer-dgmr-monsoon/resolve/main}"
ADVISORY_REPO="${INDRA_ADVISORY_REPO:-Qwen/Qwen2.5-1.5B-Instruct}"
DRY_RUN=0

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --advisory-repo) ADVISORY_REPO="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# Destinations are the paths the configuration actually reads, not a
# convention invented here:
#   inference.model.checkpoint      -> checkpoints/indra_fusion_v1.pt
#   advisory.local_advisory         -> models/advisory/Qwen2.5-1.5B-Instruct
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
ADVISORY_DIR="${REPO_ROOT}/models/advisory/$(basename "${ADVISORY_REPO}")"

# name|approximate size|destination
ARTEFACTS=(
    "indra_fusion_v1.pt|~1.4 GB|${CHECKPOINT_DIR}/indra_fusion_v1.pt"
    "earthformer_india_monsoon_17_19.pt|~0.9 GB|${CHECKPOINT_DIR}/earthformer_india_monsoon_17_19.pt"
    "dgmr_india_monsoon_17_19.pt|~0.5 GB|${CHECKPOINT_DIR}/dgmr_india_monsoon_17_19.pt"
)

echo "Indra — model weights"
echo "  source:    ${BASE_URL}"
echo "  advisory:  ${ADVISORY_REPO}"
echo

printf '%-42s %-10s %s\n' "FILE" "SIZE" "DESTINATION"
for entry in "${ARTEFACTS[@]}"; do
    IFS='|' read -r name size dest <<< "${entry}"
    printf '%-42s %-10s %s\n' "${name}" "${size}" "${dest#${REPO_ROOT}/}"
done
printf '%-42s %-10s %s\n' "${ADVISORY_REPO} (repo)" "~3.1 GB" "${ADVISORY_DIR#${REPO_ROOT}/}"
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run: nothing downloaded, no network access attempted."
    echo
    echo "Expected layout when complete:"
    echo "  checkpoints/"
    echo "    indra_fusion_v1.pt                    <- inference.model.checkpoint"
    echo "    earthformer_india_monsoon_17_19.pt    <- model.earthformer.checkpoint"
    echo "    dgmr_india_monsoon_17_19.pt           <- model.dgmr.checkpoint"
    echo "  models/advisory/$(basename "${ADVISORY_REPO}")/"
    echo "    config.json, tokenizer.json, *.safetensors"
    echo
    echo "Both directories are gitignored. /healthz reports each as unavailable"
    echo "until they are populated, and the service runs regardless."
    exit 0
fi

# Reachability first. Without this the loop below writes an HTML error page to
# indra_fusion_v1.pt, and the failure surfaces much later as an unpickling
# error that says nothing about the download.
echo "Checking ${BASE_URL} ..."
if ! curl -fsSL --head --max-time 20 "${BASE_URL}/indra_fusion_v1.pt" >/dev/null 2>&1; then
    cat >&2 <<EOF

The source is not reachable.

  ${BASE_URL}

Check the network, your access to the repository, and whether the release has
been published. Nothing was written, so re-running is safe.

Point at another source:

  bash scripts/fetch_weights.sh --base-url https://your.mirror/indra
  INDRA_WEIGHTS_BASE_URL=https://your.mirror/indra bash scripts/fetch_weights.sh

Or list the expected filenames and destinations without downloading:

  bash scripts/fetch_weights.sh --dry-run

EOF
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"
for entry in "${ARTEFACTS[@]}"; do
    IFS='|' read -r name size dest <<< "${entry}"
    if [[ -f "${dest}" ]]; then
        echo "  exists, skipping: ${dest#${REPO_ROOT}/}"
        continue
    fi
    echo "  downloading ${name} (${size})"
    # --fail so an error page is never written to the destination; a partial
    # file is removed rather than left to be loaded as a checkpoint.
    curl -fL --progress-bar --retry 3 --continue-at - \
        -o "${dest}.part" "${BASE_URL}/${name}" \
        || { rm -f "${dest}.part"; echo "  FAILED: ${name}" >&2; exit 1; }
    mv "${dest}.part" "${dest}"
done

echo
echo "Advisory model: ${ADVISORY_REPO}"
if command -v huggingface-cli >/dev/null 2>&1; then
    mkdir -p "${ADVISORY_DIR}"
    huggingface-cli download "${ADVISORY_REPO}" --local-dir "${ADVISORY_DIR}"
else
    cat >&2 <<EOF
  huggingface-cli not found. Install it and re-run, or fetch manually:

    pip install huggingface_hub
    huggingface-cli download ${ADVISORY_REPO} --local-dir ${ADVISORY_DIR}

EOF
    exit 1
fi

echo
echo "Done. Start the service and check /healthz; every component should now"
echo "report available except the NDMA collection, which is built by"
echo "'python -m indra.advisory.index_corpus'."
