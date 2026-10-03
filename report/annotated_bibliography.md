# Annotated Bibliography

**3031ICT Data Mining — Phase 2: Predicting Credit Card Default with XGBoost**
Daniel Busing · Group 24

Twelve sources, APA 7th edition, alphabetical. Each annotation states what the source
establishes, the specific decision it was used to justify in the report, and — where relevant —
where this project's evidence diverged from it.

---

Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review, 104*(3),
671–732. https://doi.org/10.15779/Z38BG31

> A legal-theoretic analysis arguing that discrimination in data mining is usually structural
> rather than intentional: models trained on historically unequal data reproduce that inequality
> even when protected attributes are excluded, because correlated features act as proxies. The
> authors' central claim for this project is that *fairness through unawareness* — deleting the
> protected column — removes the audit trail rather than the influence, and so can leave a system
> strictly harder to scrutinise. This supplied the design rationale for re-auditing the
> no-demographics arm on the same subgroups (§4.6) rather than treating that arm as a fairness
> fix. The prediction was confirmed quantitatively: removing all four demographic columns cost
> only 0.0013 ROC-AUC yet **58.0% of the male–female selection-rate gap survived** (§5.6). The
> source is normative and US-specific, so it frames the measurement but cannot settle whether any
> observed disparity is unlawful — a limitation stated explicitly in §6.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the
22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

> The originating paper for the library this report investigates, and therefore the primary source
> for the whole of §2. It contributes four things the report uses directly: the regularised
> objective with its explicit leaf-count and L1/L2 leaf-weight penalties; the second-order Taylor
> expansion in the gradient and Hessian that makes the objective quadratic per leaf and yields the
> closed-form split-gain formula reproduced in §2.1; the approximate *histogram* split-finding
> algorithm behind `tree_method="hist"` (§2.2); and the sparsity-aware split that learns a default
> direction for missing values instead of imputing them (§2.3). Reading the gain formula from this
> paper is what allowed the selected hyperparameters to be *interpreted* rather than merely
> reported — `reg_lambda` sitting in the denominator, `min_child_weight` flooring summed Hessian
> mass rather than row count (§4.1). Being a systems paper, it argues scalability, not statistical
> superiority on any given dataset; that claim is tested here empirically by the ladder.

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. In
*Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240). Association
for Computing Machinery. https://doi.org/10.1145/1143844.1143874

> Establishes the formal correspondence between ROC and precision–recall space: a curve dominating
> in one dominates in the other, but the *interpolation* between achievable points differs, and
> linear interpolation in PR space is not valid because the reachable region is non-convex. This
> governed two presentational decisions. First, PR curves in this report are drawn as attainable
> operating points rather than smoothly interpolated lines (§3.4), which matters here because the
> tuned model produces relatively few distinct scores in the high-precision region. Second, it
> supports reporting PR-AUC alongside ROC-AUC rather than instead of it: PR-AUC is the more
> sensitive instrument at 22.11% prevalence — it moves 0.3774 → 0.5602 across the ladder where
> accuracy moves barely one point (§5.1) — but it is prevalence-dependent and so unsuitable as the
> tuning objective compared across subgroups of differing base rate.

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more
correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3),
837–845. https://doi.org/10.2307/2531595

> Derives a nonparametric test for the difference between areas under ROC curves computed on the
> *same* cases, correctly accounting for the covariance that paired predictions induce — the
> standard remedy for naively comparing two AUCs as if independent. This is the exact situation of
> the final hold-out comparison, where several configurations score the identical 5,989 rows, so
> DeLong's test was applied there and only there: tuned XGBoost beats the single `PAY_0` rule by
> **+0.1399** [+0.1268, +0.1531] and library defaults by **+0.0222** [+0.0145, +0.0300] (§5.7).
> Its scope is also what made it inapplicable elsewhere. Because the test assumes one fixed sample,
> it cannot be used across cross-validation folds, whose test sets overlap between repeats; arm
> comparisons therefore report paired per-fold effect sizes against the fold-SD instead (§3.6).

Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the 17th
International Joint Conference on Artificial Intelligence* (pp. 973–978). Morgan Kaufmann.

> Shows that for a two-class problem with asymmetric error costs the expected-cost-minimising
> decision threshold has the closed form *p*\* = 1/(1 + r), where r is the ratio of false-negative
> to false-positive cost, and argues that rebalancing the training distribution is an indirect and
> inferior route to the same shift. Both results are load-bearing. The formula gives the analytic
> benchmark against which the empirically cost-minimising out-of-fold threshold is compared
> (§4.4); because Elkan's derivation *assumes true posterior probabilities*, agreement between the
> two is independent evidence of calibration rather than a tautology — and the mean absolute
> deviation duly tightens from 0.0141 to 0.0079 under isotonic recalibration (§5.4). The
> rebalancing argument underpins the rejection of `scale_pos_weight` as a decision mechanism
> (§4.3), confirmed when arms B and C′ landed 0.17% apart in cost but 34% apart in Brier score.

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art
classification algorithms for credit scoring: An update of research. *European Journal of
Operational Research, 247*(1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030

> A large-scale comparison of 41 classifiers across eight real credit-scoring datasets, and the
> standard methodological reference for the field. Two of its findings shaped this design. It
> reports AUC as the field's primary discrimination measure, which is cited in §3.4 as the first
> of three reasons for making ROC-AUC the tuning objective — the others being prevalence
> independence, needed to compare across ladder rungs and fairness subgroups, and lower
> fold-to-fold variance at a fixed trial budget. It also finds heterogeneous ensembles consistently
> near the top while individual-classifier differences are often small relative to dataset
> variation, which is the external context for this project's most important negative result: the
> entire 150-trial search spans 0.776–0.785 ROC-AUC, less than one pooled fold-SD (§4.1). Its
> datasets are proprietary and mostly European, so it motivates metric choice rather than
> predicting any absolute score on this Taiwanese sample.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb,
J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with
explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67.
https://doi.org/10.1038/s42256-019-0138-9

> Introduces TreeSHAP, a polynomial-time exact algorithm for Shapley values on tree ensembles, and
> distinguishes its two feature-perturbation schemes. This distinction drove a concrete
> configuration choice: `tree_path_dependent` weights coalitions by the split frequencies observed
> during *training*, whereas `interventional` marginalises over a supplied background
> distribution, so the latter was used with a 1,000-row stratified background sample (§4.5). The
> paper also argues that global importance is better aggregated from local attributions than read
> from built-in split statistics, which this project corroborated sharply — XGBoost's `weight`
> importance ranks `PAY_0` **twelfth** while SHAP ranks it first, so the default importance plot
> would have inverted the report's central finding (§5.5, §6). Usefully, the two schemes were
> cross-checked rather than assumed equivalent, agreeing at Spearman ρ = 0.973.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In
*Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates.

> Establishes the theoretical basis for SHAP by showing that several existing attribution methods
> are special cases of a single class, and that Shapley values are the *unique* additive
> attribution satisfying local accuracy, missingness and consistency. The uniqueness result is why
> SHAP, rather than a cheaper alternative, was chosen to answer research question 2: an
> auditability claim needs attributions that provably sum to the prediction, and §5.5 verifies this
> empirically to a maximum additivity error of **3.7 × 10⁻⁶**, float32 noise rather than method
> error. The additivity axiom also dictated working in log-odds (margin) space, where the
> decomposition is exact, and restricting probability-space attribution to the single-client
> waterfall where the sigmoid makes it non-additive (§4.5). The axioms concern faithfulness to the
> *model*, not to any causal mechanism — a limitation this report states rather than eliding, since
> a regulator may want the latter.

Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3),
239–281. https://doi.org/10.1023/A:1024068626366

> Analyses the variance of cross-validation estimates and shows that because resampled training
> sets overlap, the resulting per-fold scores are not independent; standard *t*-tests over
> repeated-CV folds therefore understate variance and are anti-conservative, declaring differences
> significant too readily. This is the methodological reason the report declines to attach
> *p*-values to arm-versus-arm comparisons, reporting paired per-fold differences against the
> fold-SD instead and refusing to call any gap under one SD an improvement (§3.6). That discipline
> did real work: it is what forced reweighting's **+0.0003** ROC-AUC movement against a 0.0071
> fold-SD to be written up as a null result (§5.3), and what keeps the +0.0035 engineered-feature
> gain from being claimed as a win (§5.2). The paper's own proposed corrected resampling test was
> not adopted, since paired effect sizes plus a single licensed DeLong test were sufficient here.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). Association
for Computing Machinery. https://doi.org/10.1145/1102351.1102430

> Characterises the calibration behaviour of common learners, showing that maximum-margin methods
> and boosted trees typically produce sigmoid-distorted probabilities pushed away from 0 and 1, and
> that Platt scaling and isotonic regression correct this — isotonic given enough data. This is the
> prior expectation against which the calibration study was designed, and it is the source the
> report *partly contradicts*: the tuned baseline arrived close to calibrated out of the box, with
> mean predicted probability 0.2189 against a 0.2212 base rate and ECE 0.0212, so isotonic
> recalibration made hold-out Brier marginally **worse** (0.1334 → 0.1344) and was not adopted
> (§5.4, §5.7). The paper's framework still explains the contrast cleanly: it is the *reweighted*
> arm that behaves as predicted, with mean prediction inflated to 0.4299 and ECE 0.2088, and that
> arm isotonic repairs to ECE 0.0080.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot
when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
https://doi.org/10.1371/journal.pone.0118432

> Demonstrates that on skewed data ROC curves can look encouraging while precision remains poor,
> because the false-positive rate is divided by a large negative count and so moves little;
> precision–recall plots expose the trade-off a practitioner actually faces. Together with Davis
> and Goadrich (2006) this is the justification for logging `aucpr` on every trial and reporting
> PR-AUC throughout while still *optimising* ROC-AUC (§3.4), and for the same cautions about PR
> interpolation. Its warning is visible in this project's own numbers: at the frozen τ = 0.1341 the
> model attains 0.8444 recall at only 0.3261 precision on the hold-out (§5.7) — an operating point
> ROC-AUC alone would not have exposed. The paper argues against using ROC as the *sole* measure
> rather than against its use as a prevalence-independent ranking objective, which is the narrower
> role it occupies here.

Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive
accuracy of probability of default of credit card clients. *Expert Systems with Applications,
36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

> The paper that introduced the dataset used throughout both phases: 30,000 Taiwanese credit-card
> clients observed over six months in 2005, with 23 explanatory variables and a next-payment
> default indicator. It compares six techniques and concludes that artificial neural networks best
> estimate the probability of default, proposing a "Sorting Smoothing Method" to validate predicted
> probabilities — an early argument that *calibration*, not just ranking, is what matters for credit
> decisions, which this report takes up through Brier score, ECE and reliability curves. It is cited
> in §1 as the source and provenance of the data, and it anchors two limitations acknowledged in §6:
> the sample is a single six-month window from one issuer, so nothing generalises without
> revalidation, and the feature set is a billing snapshot that cannot observe job loss, illness or
> income shocks — the basis for reading the plateau near ROC-AUC 0.78 as a feature-set ceiling
> rather than a model deficiency.
