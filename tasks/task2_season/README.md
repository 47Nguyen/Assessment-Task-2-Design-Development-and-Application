# Task 2 — Season Classification (`season`)

**Owner: M3** · Report section 3.2

> Predict which season a fashion item is meant for: Summer, Fall, Winter, Spring.

## Start here

```bash
cp ../_template.py train.py
```

Change `target_value = "season"` and run it:

```bash
python -m tasks.task2_season.train
```

## What the data looks like

| Class | Images | Share |
|---|---:|---:|
| Summer | 19,137 | 49.6% |
| Fall | 10,512 | 27.2% |
| Winter | 7,381 | 19.1% |
| Spring | 1,567 | 4.1% |

- 4 classes, 20 rows have no season label (dropped automatically)
- **Baseline to beat: 0.496** accuracy (always predict Summer)
- No rare-class problem — even Spring has 1,567 images

## The thing to understand before you start

This task is quick to code but probably won't produce a great score, **and that's
the expected result, not a failure.**

The season label describes which catalogue an item shipped in, not what it looks
like. A black t-shirt can be Summer or Winter — the image doesn't tell you.

Evidence for this is already in the data: Summer is the biggest category inside
*every* usage type (Casual, Sports, Ethnic, Formal). That's a merchandising
pattern, not a visual one.

> **Your goal is to prove the limit is in the label, not the model.**
> "We tried hard and it didn't work" is a weak result. "Here's evidence the
> information isn't in the image" is a strong one.

## Steps

**1. Baseline.** Always predict Summer → 0.496 accuracy. Write it down.

**2. Train the CNN.** Use `build_cnn()` from `src/models.py`, same as everyone.

**3. Try a couple of things to make the argument fair.** You need to show you
genuinely tried before concluding the label is the problem:
- A different learning rate
- A bigger network (`filters=(64, 128, 256)`)
- Class weights, since Spring is small

**4. Then build the evidence.** This is the part that earns marks:

- **Confusion matrix** — if the mistakes are spread evenly rather than
  concentrated, the model isn't finding any pattern to latch onto
- **Compare to the baseline** — how much better than 0.496 did you actually get?
- **Check if colour helps** — do dark items skew Winter and light items Summer?
  If a simple colour rule does nearly as well as the CNN, that says a lot
- **Look at examples** — find two nearly identical items with different season
  labels. One screenshot of that makes the argument instantly

## Watch out for

- Don't keep tuning forever. Once you've shown the ceiling, stop — that's the
  correct call, and you have Task 3 and the demo app to build.
- Do report accuracy **and** macro-F1. Spring is only 4% of the data, so it will
  drag macro-F1 down and that gap is informative.

## Checklist

- [ ] Baseline logged
- [ ] CNN trained and logged
- [ ] At least 2 tuning attempts logged
- [ ] Confusion matrix saved
- [ ] Evidence gathered that the label is the limit
- [ ] Model saved to `models/cnn_season.keras`
- [ ] Notes written for the report
