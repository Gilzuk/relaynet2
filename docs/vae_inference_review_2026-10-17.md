# VAE inference review — 17 October 2026

Reviewed PR #186 (`f6efe27fb61e0f31761aa6315038854696b11820`) against
main after PR #187 (`97ee4c72f7c09154efaeffa1ea38def073565927`).

The proposed correction is supported by both implementation paths in
`relaynet/relays/vae.py`: Torch `infer` decodes `mu`; NumPy `_infer` does the
same. The Torch training `forward` samples `mu + eps * std`. The evaluated
relay therefore uses the encoder's latent mean at inference, not a sampled
latent. This agrees with §4.3.3 and the retained-model audit.

The review also narrows the adjacent claim: the canonical observations do not
identify a causal effect of variational regularization or establish equivalence.
The quoted 20-dB equality is explicitly historical, with a reference to the
additional realization in §7.1.4. No data, figures, training or model code change.

Current main was merged into the existing PR branch without rewriting its
history. Conflicting PDF/bundle files were rebuilt from the combined sources;
current provenance, statistical corrections and all archived experiment evidence
are retained. The memory conflict preserves current main's state and adds this
review as a new dated entry.

Validation: numerical verifier 892 checked values / zero inconsistencies;
XeLaTeX/latexmk exit 0; 120 Arabic-numbered pages / 136 total. Chapter 1 starts
at physical page 14, body folio 120 is physical page 133, Hebrew back matter
starts at physical page 134. Revised VAE page visually inspected. No undefined
references/citations, missing characters or overfull boxes; existing font-feature
and duplicate table-anchor warnings remain. PDF, preview and source bundles are
synchronized. This is a source-backed technical review, not advisor approval.
