# Multi-seed guardrail training

Guardrail-only retrains of the paper's `dense_multi` recipe at two additional
seeds beyond the baseline seed 42 already in `runs/mit-b{0,1,2}_guard_*_s42_*/`.
Seeds 137 and 256 match the convention used by the (now-removed) ensemble
sbatch files.

Each file is a standalone 12h GPU job. Queue independently — the student SKD
checkpoint is loaded from `runs/mit-b<N>_skd_*/student_skd.ckpt` so no student
retraining happens. Output directories encode the seed so nothing collides:

```
runs/mit-b<N>_guard_dense_multi_s<seed>_j<jobid>/guardrail.ckpt
```

## Queue

```bash
sbatch slurm/multi/b0_guard_s137.sbatch
sbatch slurm/multi/b0_guard_s256.sbatch
sbatch slurm/multi/b1_guard_s137.sbatch
sbatch slurm/multi/b1_guard_s256.sbatch
sbatch slurm/multi/b2_guard_s137.sbatch
sbatch slurm/multi/b2_guard_s256.sbatch
```

6 jobs total. With a 2-job-concurrency cap that's 3 × 12h ≈ 36h wall-clock.

## Eval

After all six finish, re-run the unified eval (it picks up *newest* checkpoint
per mode per backbone; to include all three seeds, use the seed-aware eval
script instead):

```bash
sbatch slurm/eval_all.sbatch
```

`eval_all.sbatch` currently iterates newest-checkpoint-per-mode. To sweep all
seeds in the combined output, change the per-mode loop to iterate every
matching directory rather than `head -1`. One-line edit, noted below.
