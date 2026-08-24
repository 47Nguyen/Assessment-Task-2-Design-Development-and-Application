# Task 3 — Gender & Occasion Classification (`gender`, `usage`)

**Owner: M3** · Report section 3.3

> Two questions from the same image: who is this item for, and what occasion is
> it suitable for?

The assignment lets you treat these as two separate models or one combined
class. **Two separate models is simpler** — just run the same script twice with
a different target. Whichever you pick, justify it in the report.

## Start here

```bash
cp ../_template.py train_gender.py
cp ../_template.py train_usage.py
```

Set `target_value = "gender"` in one and `target_value = "usage"` in the other.

## `gender` — what the data looks like

| Class | Images | Share |
|---|---:|---:|
| Men | 20,918 | 54.2% |
| Women | 14,160 | 36.7% |
| Unisex | 2,080 | 5.4% |
| Boys | 814 | 2.1% |
| Girls | 645 | 1.7% |

- 5 classes, no missing labels
- **Baseline to beat: 0.542**

**Expect this:** Men and Women will work reasonably well. Unisex, Boys and Girls
will do badly, because:
- "Unisex" is about intent, not appearance — nothing in the image says it
- Boys/Girls items are basically adult items in smaller sizes, and you can't
  tell size from a cropped product photo

If that's what you find, it's worth arguing in the report that a
**Men / Women / Other** 3-class version is the more honest model. Test it and
compare.

## `usage` — what the data looks like

| Class | Images | Share |
|---|---:|---:|
| Casual | 29,641 | 76.8% |
| Sports | 3,940 | 10.2% |
| Ethnic | 2,570 | 6.7% |
| Formal | 2,300 | 6.0% |
| Smart Casual | 55 | 0.1% |
| Travel | 25 | 0.1% |
| Party | 13 | <0.1% |
| Home | 1 | <0.1% |

- 8 classes, 72 rows have no usage label (dropped automatically)
- **Baseline to beat: 0.769** accuracy

> **This is the clearest example of why accuracy is the wrong metric.**
>
> Always predicting "Casual" gets you **76.8% accuracy** and learns nothing.
> Meanwhile the macro-F1 ceiling is only **0.50** — if you got the four real
> classes perfect and zero on the four tiny ones, that's 4/8.
>
> Accuracy floor 0.769, macro-F1 ceiling 0.500. **Those numbers cross over.**
> That single fact is worth a paragraph in the report on its own.

**Worth testing:** merge Smart Casual / Travel / Party / Home into "Other" and
train a 5-class model. Does an honest 5-class model beat a broken 8-class one?

## Steps

For each target:

1. **Baseline** — always predict the biggest class. Log it.
2. **Train the CNN** with `build_cnn()`.
3. **Always report accuracy AND macro-F1 together.** The gap is the finding.
4. **Per-class recall table** — `per_class_report()` shows which classes get
   ignored completely.
5. **Confusion matrix** for each.
6. **Try the simplified version** — 3-class gender, 5-class usage — and compare.

## The story these two tasks tell together

Season, gender and usage all share one finding: **these labels aren't fully
recoverable from a 60×80 product photo.** Season is a catalogue decision, Unisex
is intent, and usage is 77% one class.

You own all three, so you can tell that as one connected argument rather than
three disconnected paragraphs. That's the main reason these tasks are grouped
under one person.

## Watch out for

- Never report accuracy alone on `usage`. It's misleading and a marker will
  notice.
- Don't merge the tiny classes inside `src/data.py` — keep it as an experiment
  so you can compare against the unmerged version.

## Checklist

- [ ] Baselines logged for both targets
- [ ] CNN trained and logged for both
- [ ] Accuracy and macro-F1 reported side by side
- [ ] Per-class recall tables
- [ ] Confusion matrices saved
- [ ] Simplified versions tested and compared
- [ ] Models saved to `models/cnn_gender.keras` and `models/cnn_usage.keras`
- [ ] Notes written for the report

**After this:** you'll finish before the others, so you take the demo app
(needed for HD/DI).
