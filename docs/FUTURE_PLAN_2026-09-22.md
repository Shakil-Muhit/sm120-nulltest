# Future plan after Phase 3 (proposal for the owner, 2026-09-22)

Status: Phases 0–3 are complete (`docs/EXECUTION_REPORT_2026-09-20.md`, `docs/EXECUTION_REPORT_2026-09-22.md`, `docs/contribution_test.md`). Nothing below has been started; each item is a proposal with its cost, what it would settle, and what it would not. The contribution test's rule stands: no experiment is added to rescue its verdict; the items are ordered by what they would settle for the paper as it is.

## A. Decisions that need no GPU (owner)

| item | why it matters | effort |
|---|---|---|
| A1. Clock rule at 100 ms | The registered 5 % rule, applied with the 5 Hz sampler, flags boost-clock variation (2572–2992 MHz across restarts) on the three compiled 100 ms headline cells and the first repeat of several 100 ms pairs. Options: keep the rule (the 100 ms headlines of session 3 stay clock-flagged; pair values use clock-clean repeats, as published), or amend the registration to a drift rule that compares a moving baseline rather than block-to-block boost. Either way the Sep-20 100 ms headlines (1 Hz sampler) should be re-read with the same rule for consistency. | 1 h |
| A2. Venue | Contribution test verdict: workshop or short paper, or an arXiv report with the evidence bundle; ISPASS main track would need a second model or GPU (D1) and a rate curve that can be tested rather than bracketed (C1). | decision |
| A3. Time evidence | Run the GitHub Events query with a token (`docs/provenance/push_evidence.md`) or make the repository public; run `ots upgrade docs/provenance/registration_hashes.txt.ots` after the Bitcoin attestation lands. | 15 min |
| A4. Deposit | Fill creators and license in `.zenodo.json`; deposit the subset in `docs/release/zenodo_subset.md` (session-3 evidence to be added to the manifest with the same rules). | 1–2 h |
| A5. Names | Replace the two private hostnames in live code and docs or accept them (`docs/release/release_checklist.md`). | 30 min |
| A6. Gate-08 items 114–127 | Remain deferred by the "do not nitpick" rule; list them as known audit-strictness limits in the paper's artifact section. | 0 |

## B. Analysis that needs no GPU (agent, 1–2 days)

| item | what it settles | inputs already in hand |
|---|---|---|
| B1. What freezing folds beyond the casts | X6a realized 1.40 ms at 8 s against a 0.92 ms cast ceiling (coverage check fired). Compare the C3G and C3 wrapper dumps kernel by kernel (729 vs 1,142 kernels): which non-cast kernels disappeared (weight transposes, band-split stacking, constant reshapes) and their traced time. Closes the ledger's open attribution. | `traces/dump_C3G_*`, `traces/dump_C3_*`, `ideal_costs_*.json` of session 3 |
| B2. E9 residual by term with the session-3 traces | The residual attribution (`docs/phase3/e9/e9_residual.json`) is single-term. The session-3 C3 and H traces at 8 s and 100 ms plus the Sep-20 sweep give no new lengths, so this stays a bracket; but the C3M/C3N traces show the per-band GEMM rates under templates, which bounds how much of the efficiency term is kernel selection. | session-3 traces |
| B3. Paper figures from the CSVs | `docs/phase3/figures/*.csv` are complete; render the crossover figure (model curve, sweep points, held-out points with intervals), the ledger ceiling-versus-realized bar, the E1 cast-count bars and the boundary slopes. | figure CSVs |
| B4. Sep-20 100 ms headlines under the same clock rule | Re-audit the Sep-20 evidence with `drift_flags` (1 Hz; blocks shorter than the interval use nearest samples) so the two sessions' 100 ms headlines are judged alike. | Sep-20 evidence |

## C. Measurements that would change the story (each needs a pod, owner go-ahead; ≤ 2 GPU-h each)

| item | what it would settle | what it would not |
|---|---|---|
| C1. Rate curve with enough points | Trace C3 and H at 8–10 chunk lengths between 22,050 and 352,800 (no matched pairs; node traces only, ~5 min each) to fit the per-band GEMM rate versus frames from more than two points; then re-predict the crossover and compare with the measured bracket. This is the E9 model's failing term. | Registered as a new prediction only if the held-out lengths are new; 176,400 and 264,600 are now measured and cannot serve again. |
| C2. Combined lever | One arm C3MG = repairs + max-autotune + freezing without inlining, matched against C3M at both operating points. E2 (4.8 ms) and C3G (1.4 ms) address different costs (templates versus casts); whether they add is untested and decides the best deployable configuration at 8 s. | Nothing about H; a quality gate (SDR) as before. |
| C3. Freezing on H | H + freezing without inlining at 100 ms: H's casts cost 1.0 ms of its 2.3 ms wall in E5's traces (padded fp32 weights cast per pass). If they fold, the 100 ms headline drops by up to 40 %. | The 8 s point, where H is not the best arm. |
| C4. E4b, the kernel-class RoPE/QKV epilogue | Owner's lane. The wrapper-class fold (E4a) lost 2.3 ms under max-autotune, so the X2 rows can only be realized by a kernel that writes Q, K, V in the attention layout with the scale applied in-register. | — |
| C5. Second model or second GPU | The genre bar for ISPASS/IISWC (Golden et al.; TaxBreak): one more model of the family (BS-RoFormer or a second MelBand checkpoint) or one more GPU (RTX 5080, or a data-center part) for the headline arms and E9. | The crossover on other hardware is a new registration, not a replication. |

## D. What not to do
- Do not refit the E9 cost model on the held-out points and present the refit as the result; the registration forbids it and the contribution test's value lies in the failure being reported as measured.
- Do not re-run gate 08's deferred audit-strictness items as a program; they are disclosed.
- Do not create a pod for any of C1–C5 without an explicit go-ahead in the conversation where it is created.
