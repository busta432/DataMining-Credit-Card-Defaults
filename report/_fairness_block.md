### 5.6 Fairness

**Table 6.** Subgroups at the frozen τ = 0.1341; per-group rates with Wilson intervals in B8.

| Attribute | Groups | Observed rate gap | Selection ratio, full | Ratio, no demographics | ROC-AUC gap | Recall gap |
|---|---|---|---|---|---|---|
| `SEX` | 2 | 3.44 pp | 0.855 | 0.914 | 0.003 | 0.047 |
| `EDUCATION` | 4 | 17.73 pp | **0.535** | **0.721** | 0.197 | 0.451 |
| `MARRIAGE` | 3 | 2.28 pp | 0.841 | 0.814 | 0.074 | 0.023 |
| Age band | 4 | 5.27 pp | **0.773** | 0.838 | 0.023 | 0.061 |

**The model amplifies rather than merely reflects:** a 3.44 pp observed male–female default-rate
gap becomes an **8.72 pp selection gap**, a ratio of **2.54**. Per-group ROC-AUC for `SEX` differs
by 0.003 while the false-positive-rate gap is **0.083** — equal ranking is not equal error — and
**two attributes fail the four-fifths rule** (C.2).

**Fairness through unawareness does not work here:** with all four demographic columns removed the
male–female selection gap falls only from 0.0872 to 0.0506, so **58.0% of the disparity survives a
model that cannot see sex at all**, the financial columns proxying for the demographic ones (Barocas
& Selbst, 2016).

