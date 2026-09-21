# Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC '15); with Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013)

**Reading-order item 11.** Tag: methodology STANDARD/THREAT.

## Research question
What reporting rules make a performance measurement interpretable and reproducible, and how many repetitions at which level of an experiment are needed for a confidence interval of a given width?

## Main contribution (authors' claims)
Hoefler–Belli: twelve rules. Rule 5 (quoted from the PDF): "Report if the measurement values are deterministic. For nondeterministic data, report confidence intervals of the measurement." Rule 7: compare using non-overlapping confidence intervals or ANOVA (as summarized in the sweep; the rule text is at `scratchpad/hoefler.txt`). Rule 8 (quoted): "Carefully investigate if measures such as mean or median are useful to report", with the note that nonparametric CIs of the median need more than five measurements. Survey finding: two of 95 papers report confidence intervals. Kalibera–Jones: variation exists at several levels (process executions versus iterations within an execution; execution-level variation of 30 % versus 3 % iteration-level in one example); a pilot study sets the repetition count per level; results are summarized as an effect-size interval on the ratio ("we are 95% confident that system A is faster than system B by 5.5% ± 2.5%"); 71 of the surveyed papers gave no measure of variation.

## Evidence and scope
Methodology papers: a literature survey (Hoefler–Belli) and a statistical model with DaCapo/SPEC case studies (Kalibera–Jones). No GPU content.

## Exact version and date read
Hoefler–Belli SC '15 PDF (`scratchpad/hoefler.txt`, rules at lines ~360–1561), read 2026-09-20. Kalibera–Jones ISMM 2013 PDF (`scratchpad/kalibera.txt`), extracted 2026-09-21.

## Relevant source locations
Hoefler–Belli §3 (rules), Table 1 (survey). Kalibera–Jones §3 (levels of repetition), §4 (CI of ratio), §5 (case studies).

## Relationship to this study
- The protocol already has per-level repetition (3 process restarts × 5 blocks × 50 passes, min-of-medians; `SESSION_RESULT_v2.md`, arms table header) and, since Phase 3 §2.1, block-level bootstrap intervals with 10,000 resamples on every headline and every matched pair, plus ratio intervals (`docs/phase3/stats/bootstrap_headlines.csv`, `bootstrap_pairs.csv`; e.g. C1 − C3 at 8 s +6.7412 ms [+6.6776, +6.7626], ratio 1.2326 [1.2298, 1.2344]).
- Gaps against the rules: three restarts is below the n > 5 floor for a nonparametric interval on the restart level, so the block-level interval (15 blocks per cell) is the interval reported, and E6 registers five additional restarts plus an A/A block (`prereg_amendments.md`, E6 once written). The "spread" column of Sep 20 is never called an interval.
- Scientifically meaningful: the rules are the bar, not a contribution.

## Consequence for this work
Every arm timing and every A/B carries an interval; the decision statistic for "A beats B" is the matched-pair interval excluding zero; medians are reported with the min-of-medians estimand named; a normality caveat is stated (Rule 6).

## Unresolved
Whether an ISPASS reviewer accepts block-level intervals under a deterministic ABAB alternation without the A/A calibration; E6 answers that.
