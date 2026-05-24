# Australia PII Redaction

This repository's `main` branch is now trimmed to the latest deployable hybrid model route for automatic Australian PII identification and redaction.

## Current Main Branch

- `pii-redaction-service/`: FastAPI service, web demo, API schema, policy, file input, and backend launcher.
- `hybrid-pii-model-runtime/`: minimal runtime support files for the latest hybrid backend, including label spaces, risk metadata, Qwen span-head inference code, and expected local artifact paths.
- `opf-runtime/`: OPF runtime package used by the hybrid service when `opf` is not already installed in the selected Python environment.
- `API.md`: customer-facing API description.

The recommended backend is:

```text
OPF candidate detector + Qwen 3.5 9B hard-negative span-head + risk/redaction policy
```

Start the service with:

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

## Runtime Artifacts

Large model artifacts are intentionally ignored by Git. The latest hybrid config expects:

- `hybrid-pii-model-runtime/runs/opf_hard_79/`
- `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`
- Qwen backbone at `/home/admin/model/Qwen3.5-9B-Base`, or set `REDACTION_QWEN_BACKBONE`.

## Legacy Models

Earlier model routes and training/evaluation code were removed from `main` to keep the default branch focused on the final hybrid demo. They are preserved in the `legacy-models-archive` branch.

Legacy routes preserved there include:

- `Qwen3.5_9b_base_Distill/`: earlier Qwen 9B distillation/redaction experiments, evaluation scripts, and demo API code.
- `Qwen3.5_4b_base_Full_73class/`: earlier Qwen 4B full supervised 73-class route.
- `Qwen3_4b_instruct_Distill/`: earlier Qwen 3 4B instruct distillation route.
- `opf_au_pii/`: standalone OPF training/evaluation route, AU PII label space, taxonomy configs, and OPF training utilities.
- `pii_training_prep_v3_2/`: full backend training and data preparation workspace, including OPF/Qwen span-head training, teacher data, calibration workflows, and historical reports.
- `redaction-wrapper/`: previous wrapper/service name before it was renamed to `pii-redaction-service/`.

To view the archived files:

```bash
git switch legacy-models-archive
```
