# Claim-linked evidence ledger

The evidence ledger is the authoritative map from a CausaFlux release claim to the experiment or validation artifact that supports, contradicts, or limits it.

Every row records an `evidence_id`, stable `claim_id`, status, evidence kind, source, independence/prospective flags, whether the result is synthetic, whether it represents a negative/failure, optional cycle/metric/value/threshold fields, and provenance metadata.

## Rules

- Synthetic rows are never eligible for the v2 biological release claim.
- Failed assays are never erased because they affect experiment-efficiency and cost accounting.
- Negative biological results remain in the ledger and report.
- External replication must be explicitly marked independent.
- Prospective cycle evidence must be marked prospective and identify the cycle number.
- A PASS label alone is insufficient when the evidence kind does not match the claim.
- Evidence files should be immutable or checksum-locked after evaluation.

The template is in `templates/v2_evidence/evidence_ledger.csv`.

## Provenance lock

A real PASS row qualifies for the v2 release claim only when `source` resolves to an existing local evidence artifact and `sha256` exactly matches that file. A status label without a matching locked artifact is ignored by the strict gate.
