# TRACE clinical knowledge governance

This knowledge layer is decision support, not an autonomous diagnosis or treatment system.

Every active knowledge chunk must identify one registered source, an exact section locator,
and a page locator (or `HTML-unpaginated` for web-native documents). The registered source
must have a SHA-256 hash whose scope is stated explicitly. Chunks without this provenance are
rejected before retrieval and may not be embedded.

PTB-XL SCP statements are multi-label ECG annotations. A statement describes an ECG finding;
it does not, by itself, establish a patient's symptoms, cause, prognosis, or treatment. Those
sections must either contain independently sourced clinical evidence or explicitly state that
clinical correlation is required. Management content must never be converted into a patient-
specific instruction without clinician review.

Evidence strength vocabulary: `high`, `moderate`, `limited`, `governance`.
Evidence type vocabulary: `guideline`, `scientific_statement`, `peer_reviewed_review`,
`dataset_ontology`, `internal_governance`.

