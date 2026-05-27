---
description: How YOLOmatic smart-balanced dataset splitting preserves rare classes and reproducibility.
---

# Smart Split

Smart-balanced splitting is designed for detection datasets where rare classes
can vanish from validation or test sets under a naive random split.

The current algorithm:

1. Parses image-label records and class membership.
2. Seeds target splits with rare classes first.
3. Fills remaining capacity using class balance, image-fill pressure, and seeded
   random tie-breaking.
4. Warns clearly when label files are empty and class balancing cannot be
   meaningful.

The seed changes deterministic tie-breaks, so repeated runs with the same seed
match and different seeds can produce different valid splits.

Run the preparation wizard with:

```sh
uv run yolomatic-prepare
```

Related pages: [Datasets](../guides/datasets.md), [NDJSON conversion](ndjson-conversion.md), [First training run](../getting-started/first-training-run.md).
