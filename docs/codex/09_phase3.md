Reviewed HEAD `2faa775`, including the changes made during this review.

128. **BLOCKER — E3 can reuse C1’s compilation for C1P.**  
     [fullarms.py:159](/home/muhit/ms/sm120-nulltest/fullmodel/fullarms.py:159), [matched_run.py:41](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:41). Changing global `graph_partition` does not distinguish the two compiler backends: both receive the same mode/options. Torch 2.8’s backend equality permits Dynamo to reuse C1’s compilation. A CPU counterexample reproduced this. Snapshots detect subsequent recompilation, but cannot establish that partitioning was applied; clearing counters can also make the cache-hit arm ineligible. [Torch 2.8 implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/__init__.py).  
     **Minimal fix:** include explicit arm-specific partition settings in compiler options/backend identity, merging the reduce-overhead options.

129. **BLOCKER — E1’s cast totals include substantial activation computation.**  
     [e_tools.py:60](/home/muhit/ms/sm120-nulltest/fullmodel/e_tools.py:60) includes every name containing `to_copy`, including fused normalization and attention/rotary kernels. X6a instead selects parameter-only reads at [ledger.py:256](/home/muhit/ms/sm120-nulltest/fullmodel/ledger.py:256). Applied to the existing [8-second inventory](/home/muhit/ms/sm120-nulltest/docs/phase3/ideal_costs/ideal_costs_C3_352800_v2.csv), the new match gives **505 launches and 5.197 ms**, versus X6a’s **420 launches and 0.920 ms**. This corrupts the reported cast cost and can change the contingency decision at [session3.sh:197](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:197).  
     **Minimal fix:** derive count and time from the parameter-only cast inventory used for X6a.

130. **BLOCKER — E2 counts FlashAttention as extern GEMM.**  
     [e_tools.py:74](/home/muhit/ms/sm120-nulltest/fullmodel/e_tools.py:74) matches `cutlass` anywhere in a kernel name. The existing FlashAttention name contains `cutlass::half_t`; its [inventory row](/home/muhit/ms/sm120-nulltest/docs/phase3/ideal_costs/ideal_costs_C3_352800_v2.csv:4) contributes **12 launches and 3.561 ms per pass** incorrectly to extern GEMM.  
     **Minimal fix:** exclude SDPA before GEMM classification, using the existing kernel classifier or explicit GEMM entry-point matching.

131. **BLOCKER — E4a can select a quality-ineligible C3M control.**  
     [session3.sh:246](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:246) selects C3M using `pair_eligible` and speed alone. Matched eligibility explicitly leaves SDR qualification to collection at [matched_run.py:12](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:12). Thus C3M can fail its SDR gate yet select max-autotune and become E4a’s control, contrary to E2’s fallback rule. The decision also ignores required FULLY_GRAPHED evidence. E1 similarly does not enforce its SDR kill before its branch at [session3.sh:199](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:199).  
     **Minimal fix:** join the applicable SDR and graph qualifications before selecting controls; apply C3 fallback and the E1 lane stop as registered.

132. **BLOCKER — The registered clock-drift rejection is absent.**  
     [completeness.py:230](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:230) parses clock values but discards them; acceptance at [line 266](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:266) checks timestamp coverage only. Finite clock changes exceeding the registered limit therefore leave timing comparisons eligible.  
     **Minimal fix:** associate clock values with compared blocks, enforce the registered drift threshold, and propagate rejection into measurement eligibility and session stopping.

133. **BLOCKER — Reference auditing rejects valid E9 and E5 evidence.**  
     [completeness.py:139](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:139) inventories only public operating-point references. The joins at [line 174](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:174) compare variant records against public references, while [line 184](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:184) joins held-out matched records against an inventory lacking their lengths. Both failures were reproduced in memory. Separate held-out checks later do not repair this mandatory failure.  
     **Minimal fix:** inventory pod-only E9 validation references and variant reference directories, then join by reference namespace, chunk and seed.

134. **BLOCKER — Skipping optional E5 makes mandatory completeness fail.**  
     [completeness.py:435](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:435) marks E5 traces optional, but [line 350](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:350) adds their reductions to the mandatory auxiliary list. [audit_aux](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:472) consequently rejects missing reductions after an allowed E5 skip.  
     **Minimal fix:** propagate trace optionality into auxiliary requirements.

135. **BLOCKER — Standalone session3 auditing silently requires only three headline restarts.**  
     [collect_full.py:30](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:30) defaults to three unless the shell environment supplies `REQUIRED_RESTARTS`. Although session3 exports five, a later standalone audit imports that default. [completeness.py:315](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:315) reads the configured count only for its label; acceptance uses the collector’s shorter set. Existing records qualified without r4/r5 in this configuration.  
     **Minimal fix:** derive the registered restart requirement inside collection/auditing from the selected session, independently of inherited environment.

136. **BLOCKER — An off-pod run can receive an unmarked COMPLETE verdict.**  
     [completeness.py:387](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:387) waives E9 whenever `session_pod` is false, including when `shakedown` is also false. The final record at [line 704](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:704) then retains `shakedown=false`. This permits an off-pod COMPLETE verdict without held-out evidence or an engineering-data marker.  
     **Minimal fix:** require pod mode for measurement-session completeness; explicitly classify every off-pod verdict as engineering evidence.

137. **MAJOR — E9’s held-out measurements cannot build successfully.**  
     [native_tail_adapted.py:25](/home/muhit/ms/sm120-nulltest/fullmodel/c3_fix/native_tail_adapted.py:25) permits only the earlier frame counts, and [line 31](/home/muhit/ms/sm120-nulltest/fullmodel/c3_fix/native_tail_adapted.py:31) rejects the held-out counts **401 and 601**, derived by [the registered frames formula](/home/muhit/ms/sm120-nulltest/docs/phase3/probes/e9_cost_model.py:31). Both C3 and H use this operator, so the mandatory held-out pairs fail during construction/settling.  
     **Minimal fix:** support those counts and qualify them only inside the pod session; keep shakedown qualification restricted to permitted lengths.

138. **MAJOR — Held-out A/A calibration statistics are never produced.**  
     The sole `aa_stats` invocation is at [session3.sh:155](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:155), before the held-out A/A records are generated at [line 219](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:219). E9 interpretation follows without their differences or bootstrap intervals. The audit’s aggregate check at [completeness.py:329](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:329) does not require those entries.  
     **Minimal fix:** refresh A/A statistics before interpreting E9 and require the corresponding held-out entries on pod sessions.

EXECUTION-READY: no

- Verification used source inspection, existing traces and in-memory CPU fixtures. No repository files were edited, and no GPU measurements or held-out inputs were executed.
- C1P’s requested skip-reason logging depends on inherited `TORCH_LOGS`: [session_lib.sh:28](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:28) omits the new arms.
tokens used
231,247
Reviewed HEAD `2faa775`, including the changes made during this review.

128. **BLOCKER — E3 can reuse C1’s compilation for C1P.**  
     [fullarms.py:159](/home/muhit/ms/sm120-nulltest/fullmodel/fullarms.py:159), [matched_run.py:41](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:41). Changing global `graph_partition` does not distinguish the two compiler backends: both receive the same mode/options. Torch 2.8’s backend equality permits Dynamo to reuse C1’s compilation. A CPU counterexample reproduced this. Snapshots detect subsequent recompilation, but cannot establish that partitioning was applied; clearing counters can also make the cache-hit arm ineligible. [Torch 2.8 implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/__init__.py).  
     **Minimal fix:** include explicit arm-specific partition settings in compiler options/backend identity, merging the reduce-overhead options.

129. **BLOCKER — E1’s cast totals include substantial activation computation.**  
     [e_tools.py:60](/home/muhit/ms/sm120-nulltest/fullmodel/e_tools.py:60) includes every name containing `to_copy`, including fused normalization and attention/rotary kernels. X6a instead selects parameter-only reads at [ledger.py:256](/home/muhit/ms/sm120-nulltest/fullmodel/ledger.py:256). Applied to the existing [8-second inventory](/home/muhit/ms/sm120-nulltest/docs/phase3/ideal_costs/ideal_costs_C3_352800_v2.csv), the new match gives **505 launches and 5.197 ms**, versus X6a’s **420 launches and 0.920 ms**. This corrupts the reported cast cost and can change the contingency decision at [session3.sh:197](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:197).  
     **Minimal fix:** derive count and time from the parameter-only cast inventory used for X6a.

130. **BLOCKER — E2 counts FlashAttention as extern GEMM.**  
     [e_tools.py:74](/home/muhit/ms/sm120-nulltest/fullmodel/e_tools.py:74) matches `cutlass` anywhere in a kernel name. The existing FlashAttention name contains `cutlass::half_t`; its [inventory row](/home/muhit/ms/sm120-nulltest/docs/phase3/ideal_costs/ideal_costs_C3_352800_v2.csv:4) contributes **12 launches and 3.561 ms per pass** incorrectly to extern GEMM.  
     **Minimal fix:** exclude SDPA before GEMM classification, using the existing kernel classifier or explicit GEMM entry-point matching.

131. **BLOCKER — E4a can select a quality-ineligible C3M control.**  
     [session3.sh:246](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:246) selects C3M using `pair_eligible` and speed alone. Matched eligibility explicitly leaves SDR qualification to collection at [matched_run.py:12](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:12). Thus C3M can fail its SDR gate yet select max-autotune and become E4a’s control, contrary to E2’s fallback rule. The decision also ignores required FULLY_GRAPHED evidence. E1 similarly does not enforce its SDR kill before its branch at [session3.sh:199](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:199).  
     **Minimal fix:** join the applicable SDR and graph qualifications before selecting controls; apply C3 fallback and the E1 lane stop as registered.

132. **BLOCKER — The registered clock-drift rejection is absent.**  
     [completeness.py:230](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:230) parses clock values but discards them; acceptance at [line 266](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:266) checks timestamp coverage only. Finite clock changes exceeding the registered limit therefore leave timing comparisons eligible.  
     **Minimal fix:** associate clock values with compared blocks, enforce the registered drift threshold, and propagate rejection into measurement eligibility and session stopping.

133. **BLOCKER — Reference auditing rejects valid E9 and E5 evidence.**  
     [completeness.py:139](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:139) inventories only public operating-point references. The joins at [line 174](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:174) compare variant records against public references, while [line 184](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:184) joins held-out matched records against an inventory lacking their lengths. Both failures were reproduced in memory. Separate held-out checks later do not repair this mandatory failure.  
     **Minimal fix:** inventory pod-only E9 validation references and variant reference directories, then join by reference namespace, chunk and seed.

134. **BLOCKER — Skipping optional E5 makes mandatory completeness fail.**  
     [completeness.py:435](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:435) marks E5 traces optional, but [line 350](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:350) adds their reductions to the mandatory auxiliary list. [audit_aux](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:472) consequently rejects missing reductions after an allowed E5 skip.  
     **Minimal fix:** propagate trace optionality into auxiliary requirements.

135. **BLOCKER — Standalone session3 auditing silently requires only three headline restarts.**  
     [collect_full.py:30](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:30) defaults to three unless the shell environment supplies `REQUIRED_RESTARTS`. Although session3 exports five, a later standalone audit imports that default. [completeness.py:315](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:315) reads the configured count only for its label; acceptance uses the collector’s shorter set. Existing records qualified without r4/r5 in this configuration.  
     **Minimal fix:** derive the registered restart requirement inside collection/auditing from the selected session, independently of inherited environment.

136. **BLOCKER — An off-pod run can receive an unmarked COMPLETE verdict.**  
     [completeness.py:387](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:387) waives E9 whenever `session_pod` is false, including when `shakedown` is also false. The final record at [line 704](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:704) then retains `shakedown=false`. This permits an off-pod COMPLETE verdict without held-out evidence or an engineering-data marker.  
     **Minimal fix:** require pod mode for measurement-session completeness; explicitly classify every off-pod verdict as engineering evidence.

137. **MAJOR — E9’s held-out measurements cannot build successfully.**  
     [native_tail_adapted.py:25](/home/muhit/ms/sm120-nulltest/fullmodel/c3_fix/native_tail_adapted.py:25) permits only the earlier frame counts, and [line 31](/home/muhit/ms/sm120-nulltest/fullmodel/c3_fix/native_tail_adapted.py:31) rejects the held-out counts **401 and 601**, derived by [the registered frames formula](/home/muhit/ms/sm120-nulltest/docs/phase3/probes/e9_cost_model.py:31). Both C3 and H use this operator, so the mandatory held-out pairs fail during construction/settling.  
     **Minimal fix:** support those counts and qualify them only inside the pod session; keep shakedown qualification restricted to permitted lengths.

138. **MAJOR — Held-out A/A calibration statistics are never produced.**  
     The sole `aa_stats` invocation is at [session3.sh:155](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:155), before the held-out A/A records are generated at [line 219](/home/muhit/ms/sm120-nulltest/scripts/session3.sh:219). E9 interpretation follows without their differences or bootstrap intervals. The audit’s aggregate check at [completeness.py:329](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:329) does not require those entries.  
     **Minimal fix:** refresh A/A statistics before interpreting E9 and require the corresponding held-out entries on pod sessions.

EXECUTION-READY: no

- Verification used source inspection, existing traces and in-memory CPU fixtures. No repository files were edited, and no GPU measurements or held-out inputs were executed.
- C1P’s requested skip-reason logging depends on inherited `TORCH_LOGS`: [session_lib.sh:28](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:28) omits the new arms.
