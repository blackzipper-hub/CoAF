# Casual CoAF Causal I2V Training

## Quick start

```bash
# Submit training jobs (priority order)
sbatch jobs/train/i2v/train_causal_v4_depth_rgb_480640_5k_60k.sbatch  # P1
sbatch jobs/train/i2v/train_causal_v4_depth_rgb_256_5k_60k.sbatch     # P2
sbatch jobs/train/i2v/train_causal_v1_pose_rgb_5k_60k.sbatch          # P3
sbatch jobs/train/i2v/train_causal_v5_pose_depth_rgb_5k_60k.sbatch    # P4
sbatch jobs/train/i2v/train_causal_v2_flow_rgb_5k_60k.sbatch          # P5
```

Reuses CoAF finetrainers + `CogVideoX-5b-I2V` via absolute paths. All jobs: causal attention, 60k steps, checkpoint every 5k.

## Training hyperparameters (8× H800)

| Parameter | Value |
|-----------|-------|
| per-GPU batch | 1 |
| GPU count | 8 (data parallel via `accelerate launch`) |
| gradient accumulation | 8 |
| **Effective batch size** | 1 × 8 × 8 = **64** |

Override via env: `NUM_GPUS`, `TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`.

## Resolution note (480×640)

**`Casual_CoAF/coaf_dataset/raw` does not contain native 480×640 video.**

- Raw RGB PNGs are **256×256** (written by `preprocess_raw.py` from Bridge V Full).
- Bridge TFDS `image_0` is also **256×256** — there is no higher-resolution source in the current pipeline.
- `v4_depth_rgb_480640` is produced by **upscaling** 256→640×480 in `compose_all.py` (LANCZOS4). It will look softer than true HD; this is a data limit, not a compose bug.

To get sharper 480×640 data you would need a new upstream render/export at that resolution (outside current Bridge 256 pipeline).

## Data preparation

```bash
# Full 480×640 depth+rgb compose (5000 episodes)
sbatch ../coaf_dataset/scripts/jobs/compose_v4_depth_rgb_480640.sbatch

# Rebuild manifest after partial compose
python ../coaf_dataset/scripts/rebuild_composed_manifest.py \
  ../coaf_dataset/composed/v4_depth_rgb_480640
```

## Verify before training

```bash
bash scripts/verify_training_setup.sh
```
