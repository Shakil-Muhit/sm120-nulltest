# NVIDIA TensorRT documentation, "Dynamic Shapes: Core Concepts" (TensorRT 11.3.0, read 2026-09-22)

**Contribution-test source (shape-specialized implementation selection in a production inference engine).**

## Research question
Not a study. Vendor documentation of how an inference engine chooses kernels ("tactics") when input dimensions vary.

## Main contribution (vendor text, verbatim where quoted)
"An *optimization profile* describes a range of dimensions for each network input and the dimensions the auto-tuner will use for optimization." Tactics are chosen by the builder's auto-tuner for the profile's optimization dimensions and must be valid over the profile's [minimum, maximum] range; "Some operations have tactics that only work for `MIN=OPT=MAX`, so when these values differ, the tactic is disabled." The page does not describe switching kernels at runtime by the actual input size within one profile, and gives no guidance on performance away from the optimization dimensions. (That the builder times candidate tactics and keeps the fastest is the documented behaviour of the TensorRT builder; the exact developer-guide section was not opened for this note and the statement is made at that level of certainty.)

## Evidence and scope
Documentation; no measurements.

## Exact version and date read
docs.nvidia.com, TensorRT 11.3.0 ("Latest"), page "Dynamic Shapes: Core Concepts"; read 2026-09-22. The 10.16 glossary and the 8.2.5 archive pages were not reachable.

## Relationship to this study
- Overlap: implementation choice depends on input size and is decided by timing at a chosen point; profiles are the engine's answer to the operating-point question this study frames as 100 ms versus 8 s.
- Difference: selection is empirical (auto-tuner timing at OPT) and static per profile; no attributed cost model, no statement of where in a range one implementation overtakes another, no held-out test. This study's E9 registers such a model and tests it at lengths it was not built on; the model failed the tolerance and the crossover was bracketed by measurement.
- Scientifically meaningful: the practice of shape-specialized selection is established; an attributed, pre-registered crossover model for a band-split audio forward, and its held-out failure, were not found in the sources reviewed.

## Consequence for this work
The contribution test cites TensorRT profiles as the deployed form of size-dependent selection; the study claims neither the practice nor a working predictive model, only the attributed test and its outcome.

## Unresolved
The exact builder-section wording on tactic timing at TensorRT 11.3.
