# Indra

Hyper-localised precipitation nowcasting over India, with attribution that a
forecaster can read.

Six hours ahead, at 30-minute steps, on a ~9 km grid covering 6–38°N and
68–100°E. Every forecast comes with an ensemble spread, a district-level
hazard probability, and an explanation of which inputs moved it.

## Artefacts

Weights, climatology and boundaries are versioned outside git and pulled in —
several gigabytes of binary that change on every retrain do not belong in a
source tree:

| Artefact | Fetch with |
| --- | --- |
| Model checkpoints | `scripts/fetch_weights.sh` |
| Climatological statistics | `scripts/fetch_data.sh` |
| District boundaries, static priors | `scripts/fetch_data.sh` |
| Advisory model | `scripts/fetch_weights.sh` |
| NDMA guideline corpus | `python -m indra.advisory.index_corpus` |
| INSAT / IMDAA / IMD archives | MOSDAC and IMD |

The service starts before they arrive. `/healthz` reports per-component
readiness and endpoints return 503 naming the specific artefact, so a
deployment can be brought up incrementally rather than debugging a boot
failure.

Nothing substitutes for a missing input. A missing climatology raises rather
than falling back to batch statistics — those leak across the temporal split;
a missing corpus stops the advisory rather than letting the model invent
mitigation guidance.

## Architecture

```
INSAT-3D/3DR (HDF5) ─┐
IMDAA (GRIB2)        ├─► ingestion + QC ─► reprojection ─► normalisation ─► (13, 30, 384, 384)
IMD surface (.grd)   │
static priors (TIFF) ┘
                                                                    │
                          ┌─────────────────────────────────────────┘
                          ▼
              Earthformer backbone          space-time cuboid attention
                          │                 (B,13,30,384,384) → (B,12,96,96,128)
                          ▼
                   adapter bridge           latent → conditioning pyramid
                          │                 [(96,96²),(192,48²),(384,24²),(768,12²)]
                          ▼
                  DGMR generator            ConvGRU sampler, seeded noise
                          │                 → (B,12,1,384,384) mm/h
                          ▼
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ensemble stats    Captum IG +       district extraction
   + thresholds      attention maps    + CAP 1.2 advisory
```

A deterministic transformer predicts *where* rain will be; a generative head
restores the sharp convective structure that an L1/L2-trained model blurs
away. The adapter is what lets the two compose at all — Earthformer emits a
per-lead-time latent, DGMR wants a multi-scale spatial conditioning stack.

Ground truth is INSAT-3D/3DR HEM, a half-hourly rain-rate retrieval. The IMD
gauge grid is a *daily* accumulation and cannot supervise a 30-minute forecast:
held across 48 slots it would make every lead frame in a day identical.

## Inputs

30 channels, in fixed order. The order is a contract — the XAI layer reports
"850 hPa relative humidity", not "channel 14".

| Index | Source | Channels |
| --- | --- | --- |
| 0–2 | INSAT-3D/3DR L1C/L2B | TIR1, WV, cloud-top temperature |
| 3–22 | IMDAA reanalysis | u, v, t, rh, z at 200/500/700/850 hPa |
| 23–26 | IMD surface | precipitation, 2 m temperature, 2 m RH, 10 m wind |
| 27–29 | Static priors | Cartosat DEM, Bhuvan LULC, ICAR soil |

Thirteen frames of history at 30 minutes (t−6h to t0), twelve lead frames out
to +6h.

## Training and validation split

Chronological, never random: consecutive 30-minute frames of one storm are
nearly identical, so a random split scores the model on data it has seen.

Train on JJAS 2017–2019, validate on JJAS 2020. The split is tested at each
window's *reaching edge* — a training window's last target frame must land
before the cutoff, and a validation window's earliest input frame after it —
because a window spans six hours in both directions and testing `t0` alone
would let a training target and a validation input describe the same storm.

The climatological statistics used for normalisation are fitted on 2017–2019
only. Statistics fitted on the held-out season would carry its distribution
into every normalised field, and no shape check would notice.

## Layout

```
configs/            five YAML files, cross-validated against each other
  data/             ingestion, preprocessing
  model/            fusion relay
  train/            training
  inference/        serving, XAI, advisory, API

src/indra/
  ingestion/        one reader per stream, + QC (dropout, bounds, parallax, gap-fill)
  preprocessing/    reprojection, normalisation, tensor assembly
  models/           earthformer, adapter, dgmr, fusion
  datasets/         window enumeration, targets, disk cache, Dataset
  training/         losses, replay buffer, trainer
  evaluation/       CSI, POD, FAR, MAE, CRPS
  xai/              baselines, integrated gradients, attention, report
  advisory/         district extraction, RAG retrieval, CAP synthesis
  api/              FastAPI service

frontend/           React + Vite (vendored, unmodified)
scripts/            train.py, evaluate.py, ct_promote.py, asset fetchers
tests/              window split, grid geometry, metrics, gate, config
.github/workflows/  ci.yml (lint, types, imports, tests), ct.yml (retrain cycle)
```

## Getting started

```bash
cp .env.example .env
docker compose up
```

Backend on `:8000`, frontend on `:5173`. Both start without weights; check
`/healthz` to see what is missing.

To run the backend directly:

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn indra.api.main:app --reload
```

Fetching artefacts — `--dry-run` prints the targets and the expected on-disk
layout without touching the network:

```bash
bash scripts/fetch_data.sh --dry-run
bash scripts/fetch_data.sh
bash scripts/fetch_weights.sh
python -m indra.advisory.index_corpus --create-sample
```

Both scripts take `--base-url` (or `INDRA_DATA_BASE_URL` /
`INDRA_WEIGHTS_BASE_URL`) to pull from a mirror or an internal bucket.

## API

| Endpoint | Returns |
| --- | --- |
| `GET /api/forecast/point?lat=&lon=` | 12-frame rain rate and exceedance probability at one grid cell |
| `GET /api/advisories/districts` | Qualifying districts with their CAP 1.2 alerts |
| `GET /api/xai/explanation` | Evidence frames, channel drivers, attribution and attention maps |
| `GET /healthz` | Readiness, per-component status, grid and domain metadata |

A nowcast is computed once per valid time and cached, so a point query is an
array index rather than a forward pass. Concurrent requests for the same valid
time share one computation.

## Explainability

Integrated Gradients over the input window against a **climatological mean**
baseline, not zeros. Twenty-three of the thirty channels are z-scored, where
the climatological mean is 0.0, so zeros would be accidentally right for most
of the tensor and badly wrong for the rest — normalised zero on `insat_tir1`
is 180 K, an impossibly cold cloud top over the whole subcontinent, and every
attribution would be dominated by that difference.

The attribution target is the exceedance probability, relaxed to
`sigmoid((y − 15)/T)` so it has a gradient. The API keeps reporting the true
counted probability; the relaxation exists only to carry gradients, and its
width is recorded in every report.

Attention maps and evidence frames come from the Earthformer's encoder and
decoder cross-attention. Attention shares are reported against the uniform
baseline (1/13 = 7.7%), because "8% attention on frame t−5" reads as a finding
and is exactly what indifference looks like.

## Advisories

District extraction intersects the exceedance field with administrative
boundaries, then retrieval pulls NDMA guidance scoped to the hazard class,
severity tier and terrain. A local model composes the CAP 1.2 alert under
guided decoding, so the output is schema-valid by construction rather than
parsed and repaired.

Everything runs in-process. No API key, no external endpoint, no request
carrying a district's forecast to a third party.

Four constraints the generation cannot escape:

- Severity may be raised above the forecast band, never lowered.
- Certainty may be lowered below what the ensemble supports, never raised.
- `Observed` certainty is rejected. This is a forecast.
- With no retrieved guidance, the model is confined to generic advice and
  forbidden from inventing specific protocols. The alert reports
  `grounded_in_ndma: false`.

## Training

```bash
python scripts/train.py --dry-run          # build everything, train nothing
python scripts/train.py
python scripts/train.py --max-steps 1000 --device cpu
```

Enumerates windows, splits at the reaching edge, builds the loaders, loads the
relay and its critics, resumes the replay buffer from its manifest, and runs
the adversarial loop. Requires the checkpoints and the climatology; both print
what is missing and exit rather than ending in a traceback.

## Continuous training

```bash
python scripts/evaluate.py --checkpoint checkpoints/candidate.pt \
                           --out metrics/candidate.json
python scripts/ct_promote.py --candidate metrics/candidate.json \
                             --incumbent metrics/production.json
```

A cycle is fetch, train, evaluate on the held-out season, gate, promote.
`.github/workflows/ct.yml` runs it monthly and on demand.

The gate lives in `indra/evaluation/gate.py` rather than in the script, so the
rule that decides what reaches production is unit-tested. A candidate promotes
only if it improves the monitored metric by a margin, regresses no guarded
metric, and was scored over enough observed events — that last condition
matters most, because heavy rain is rare enough that a validation pass can
produce a CSI computed from four events, and without a floor a quiet fortnight
promotes on noise.

`ct_promote.py` exits 0 on promotion and 1 on rejection, and copies nothing
without `--apply`.

## Verification

```bash
pip install -r requirements-dev.txt
ruff check src scripts tests
ruff format --check src scripts tests
mypy
python scripts/verify_imports.py
pytest
```

`verify_imports.py` imports all 47 modules in dependency order and loads the
five configs through the cross-file validator. `pytest` runs 102 tests covering
what runs without data: window enumeration and the split rule, grid geometry
and cell area, the verification metrics, the promotion gate, and the
configuration validators.
Neither needs weights, a corpus or a GPU.

Anything requiring a granule, a checkpoint or the NDMA collection is out of
scope for the suite rather than mocked. A test against fabricated inputs
asserts that the fabrication behaves as expected, which is not a property of
this system.

CI runs all of the above plus a Docker build on every push and pull request.

## Licence

MIT. See `LICENSE`.
