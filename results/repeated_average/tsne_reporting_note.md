`Fig3_tsne_repeated_pooled.*` is a repeated-evaluation visualization built from the pooled test embeddings across all repeated Route B runs.

A literal arithmetic average of t-SNE coordinates is not statistically well-defined, because each t-SNE embedding is a nonlinear low-dimensional projection with run-specific geometry. For that reason, the new AE9 package reports:

- averaged quantitative metrics via `mean ± std`;
- a pooled repeated-evaluation t-SNE as the qualitative companion figure.

This keeps the original AE7-copy method and code logic unchanged while avoiding a misleading "averaged t-SNE coordinate" claim.
