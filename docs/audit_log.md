# Audit log

Owner audit events and stamp re-anchors. Entries are the owner's words verbatim;
verification records beneath each are Code's.

## 2026-09-01 — traces.sh stamp re-anchor

> traces.sh: on-disk copy found corrupted (self-paste at :131, mtime Aug 31,
> pre-session; bash -n failed → could never have produced numbers). Reconstructed
> by Code from embedded complete copy; verified by diff against the audited
> paste of Sep 1 — sole deltas are the item-2 floor guard. Stamp re-anchored.

> And say the quiet part in the log too, because it's tonight's thesis made
> flesh: a zero-commit repo has no way to even notice this. One push and
> git status screams on the next stray paste-into-terminal; git checkout
> restores it in two seconds. The corruption is not a reason for alarm — it's
> the last argument the push needed.

Verification record (Code): audited reference reconstructed as shebang + the
embedded complete copy (pre-fix snapshot lines 131-291 — the byte-identical
source of the paste displayed and audited in-session); 162 lines, parses,
sha256 9d1e0a2d6f72a1b6…; fingerprint lines 1/82/95/117/161 match the paste
verbatim. `diff traces_audited.sh analysis/traces.sh` → exactly one hunk:
the 17-line item-2 graph-trace-failure guard (146a147,163). No other delta.

## 2026-09-01 — arm E tolerance tier corrected

> Reclassify E → tight tier (rtol=1e-4/atol=1e-5) per the classification rule
> in the rationale comment: repo-E contains no Inductor-generated kernels.
> Loose tier is now C1/C2 only. Keep the never-validated note on C1, C2, and
> E — all three are pod-opening validation items. If the amended prereg later
> redefines E as an Inductor arm, the build branch and the tier move together
> under the same rule — flag it then.

Verification record (Code): `VAL_TOL` now tight = A/B/D/E/F/G, loose = C1/C2
(AST-checked, all 8 arms covered); `tolerance_class` in the run JSON follows;
README Deviations item 4 updated. The rule outranked the ledger — the loose-E
assignment came from the owner's Vol. 1 arm-ledger reconstruction; the file as
built decided.
