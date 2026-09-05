#!/usr/bin/env bash
# =============================================================================
# Fetch data artefacts: climatology, district boundaries, sample granules.
#
#   bash scripts/fetch_data.sh --dry-run      # print targets, no network
#   bash scripts/fetch_data.sh                # download
#   bash scripts/fetch_data.sh --base-url https://your.mirror/monsoon-data
#
# Override the source with --base-url or INDRA_DATA_BASE_URL to pull from a
# mirror, an internal bucket or a local path. The raw INSAT, IMDAA and IMD
# archives are obtained from MOSDAC and IMD directly; only the derived
# artefacts and a one-day sample are served here.
#
# One artefact deserves more care than the rest.
# configs/climatology/monsoon_17_19_stats.json holds the channel means and
# standard deviations every input is normalised against, and the attribution
# baseline is derived from it. It is fitted on JJAS 2017-2019 only, excluding
# the 2020 validation season on purpose: statistics fitted on held-out data
# carry that season's distribution into every normalised field, and no shape
# check anywhere would notice. Do not substitute a file computed over a
# different period, and do not compute one from whatever data happens to be
# on disk.
#
# Nothing here runs automatically.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${INDRA_DATA_BASE_URL:-https://huggingface.co/datasets/indra-nowcasting/monsoon-data/resolve/main}"
DRY_RUN=0

usage() {
    sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# remote path|approximate size|destination, all read from configuration:
#   preprocessing.normalization.stats_path
#   inference.geospatial.admin_boundaries
#   ingestion.sources.*.root
ARTEFACTS=(
    "climatology/monsoon_17_19_stats.json|~40 KB|${REPO_ROOT}/configs/climatology/monsoon_17_19_stats.json"
    "boundaries/india_admin_district.gpkg|~28 MB|${REPO_ROOT}/data/static/india_admin_district.gpkg"
    "static/cartosat_dem_india.tif|~310 MB|${REPO_ROOT}/data/static/cartosat_dem_india.tif"
    "static/bhuvan_lulc_india.tif|~95 MB|${REPO_ROOT}/data/static/bhuvan_lulc_india.tif"
    "static/icar_soil_india.tif|~72 MB|${REPO_ROOT}/data/static/icar_soil_india.tif"
    "sample/insat3d_20200920.tar|~1.8 GB|${REPO_ROOT}/data/raw/insat3d/insat3d_20200920.tar"
    "sample/insat3d_hem_20200920.tar|~240 MB|${REPO_ROOT}/data/raw/insat3d_hem/insat3d_hem_20200920.tar"
)

echo "Indra — data artefacts"
echo "  source: ${BASE_URL}"
echo

printf '%-48s %-10s %s\n' "FILE" "SIZE" "DESTINATION"
for entry in "${ARTEFACTS[@]}"; do
    IFS='|' read -r name size dest <<< "${entry}"
    printf '%-48s %-10s %s\n' "${name}" "${size}" "${dest#${REPO_ROOT}/}"
done
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
    cat <<EOF
Dry run: nothing downloaded, no network access attempted.

Expected layout when complete:
  configs/climatology/
    monsoon_17_19_stats.json        <- JJAS 2017-2019 ONLY; see the header
  data/static/
    india_admin_district.gpkg       <- inference.geospatial.admin_boundaries
    cartosat_dem_india.tif          <- dem_elevation
    bhuvan_lulc_india.tif           <- lulc_mask
    icar_soil_india.tif             <- soil_type
  data/raw/insat3d/                 <- ingestion.sources.insat.root
  data/raw/insat3d_hem/             <- ingestion.targets.hem.root

Obtained separately:
  The NDMA guideline corpus. Build the collection from source documents with
  'python -m indra.advisory.index_corpus', or '--create-sample' for a labelled
  bootstrap corpus that exercises the pipeline before the authoritative one.

  The full JJAS 2017-2020 record. The sample above is one day, enough to
  exercise ingestion; training needs the archive from MOSDAC and IMD.
EOF
    exit 0
fi

echo "Checking ${BASE_URL} ..."
if ! curl -fsSL --head --max-time 20 \
        "${BASE_URL}/climatology/monsoon_17_19_stats.json" >/dev/null 2>&1; then
    cat >&2 <<EOF

The source is not reachable.

  ${BASE_URL}

Check the network, your access to the dataset repository, and whether the
release has been published. Nothing was written, so re-running is safe.

This script will not synthesise a climatology in its place: invented channel
statistics would bias every field the model sees, silently.

Point at another source:

  bash scripts/fetch_data.sh --base-url https://your.mirror/monsoon-data

Or inspect the expected layout without downloading:

  bash scripts/fetch_data.sh --dry-run

EOF
    exit 1
fi

for entry in "${ARTEFACTS[@]}"; do
    IFS='|' read -r name size dest <<< "${entry}"
    if [[ -f "${dest}" ]]; then
        echo "  exists, skipping: ${dest#${REPO_ROOT}/}"
        continue
    fi
    mkdir -p "$(dirname "${dest}")"
    echo "  downloading ${name} (${size})"
    curl -fL --progress-bar --retry 3 --continue-at - \
        -o "${dest}.part" "${BASE_URL}/${name}" \
        || { rm -f "${dest}.part"; echo "  FAILED: ${name}" >&2; exit 1; }
    mv "${dest}.part" "${dest}"
done

echo
echo "Done. Next:"
echo "  bash scripts/fetch_weights.sh"
echo "  python -m indra.advisory.index_corpus --create-sample"
