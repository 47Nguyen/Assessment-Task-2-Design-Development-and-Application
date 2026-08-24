# Task 4 — Visual Search

**Owner: M4** · Report section 3.4

> Given a query image, find the items in the catalogue that look most similar to
> it. Return the top K.

This is different from the other three: there are no classes to predict. You're
building a search engine over ~38,600 images.

## Start here

```bash
cp ../_template.py search.py
```

You don't need a target — use `get_images_only()` instead of `get_split()`:

```python
from src.data import get_images_only

X, meta = get_images_only()   # all 38,612 images + their metadata
```

## How it works

Every image gets turned into a list of numbers (an "embedding"). Similar-looking
images end up with similar numbers. To search, you embed the query image and
find the closest ones.

```
image -> [0.2, -1.4, 0.8, ...] -> compare against all 38k -> return closest 5
```

The whole task is: **what's the best way to turn an image into those numbers?**

## Build it as a ladder

Do these in order. Each one should be better than the last, and comparing them
is what makes this a real investigation rather than one model with no context.

| Rung | Method | Needs Task 1? |
|---|---|---|
| 1 | Flatten the raw pixels | No |
| 2 | PCA on the pixels (e.g. down to 128 numbers) | No |
| 3 | Autoencoder — train a network to compress and rebuild images, use the middle layer | No |
| 4 | Take the CNN from Task 1 and use its second-to-last layer | **Yes** |
| 5 | Train with triplet loss (advanced, optional) | Yes |

**Start with rungs 1–3 in Week 2** — they don't need anyone else's work.
Rung 4 needs M1's trained model, which arrives around Week 3.

Rung 1 will be bad. That's the point — it gives you a floor to measure against.

## Finding the closest images

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(embeddings)                    # embeddings: (38612, 128)
distances, indices = nn.kneighbors(query_embedding)
```

38,612 images is small. Plain `NearestNeighbors` is fast enough — you don't need
FAISS or anything fancy, and saying so in the report is a good justification.

## Measuring how good it is

There's no "correct answer" for similarity, so use the labels as a stand-in:
**if a retrieved item has the same `articleType` as the query, count it as
relevant.**

**Precision@K** — of the top K results, what fraction matched?

```python
# for each query, check if retrieved items share the query's articleType
hits = retrieved_labels == query_label
precision_at_5 = hits[:, :5].mean()
```

Report it two ways:
- **Strict** — same `articleType` (Flip Flops must return Flip Flops)
- **Lenient** — same `subCategory` (Sandals counts for a Flip Flops query)

Strict alone is unfair. A shopper looking at flip flops would be perfectly happy
to see sandals.

**Also show pictures.** Pick 5 query images, show the top 5 results for each in a
grid. Bad retrieval is obvious to the eye in a way the numbers can miss, and it
makes a great report figure.

## Steps

1. Load all images with `get_images_only()`
2. Rung 1: flatten pixels → build index → measure Precision@K → save the number
3. Rung 2: PCA → same measurements
4. Rung 3: autoencoder → same measurements
5. **Wait for Task 1** → rung 4 → same measurements
6. Build the comparison table across all rungs
7. Make the query→top-5 image grid

## While you're waiting for Task 1

Use Week 3 to build the evaluation code and the image grid function. Then when
M1's model lands, rung 4 is a five-minute job because everything else is ready.

## Watch out for

- **Don't apply colour jitter** if you add augmentation anywhere. Colour is a big
  part of visual similarity — changing it breaks the thing you're measuring.
- **Exclude the query itself** from its own results, or Precision@1 is always 1.0.
- **Normalise embeddings** before cosine similarity, or magnitude dominates.

## Checklist

- [ ] Rung 1 (raw pixels) working and measured
- [ ] Rung 2 (PCA) working and measured
- [ ] Rung 3 (autoencoder) working and measured
- [ ] Rung 4 (Task 1 CNN features) working and measured
- [ ] Comparison table across all rungs
- [ ] Query → top-5 visual grid figure
- [ ] Embeddings saved to `models/embeddings.npy`
- [ ] Notes written for the report
