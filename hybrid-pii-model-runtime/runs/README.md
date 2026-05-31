# Runtime Model Artifacts

This directory is reserved for local model artifacts used by the latest hybrid
backend.

Expected paths:

- `opf_hard_79/`: OPF checkpoint directory.
- `qwen9b_hn_spancls_heads/last_linear/head.pt`: Qwen 3.5 9B hard-negative span-head checkpoint.

For artifact sizes, checksums, external model dependencies, and verification
commands, see:

```text
../../docs/ARTIFACT_MANIFEST.md
```

The service config at
`../../pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json` points
here by default. Override with `REDACTION_PII_PROJECT_ROOT` if artifacts live
elsewhere.
