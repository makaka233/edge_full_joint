# R12-A 100GS failure: forensic diagnosis and bounded correction

**Evidence source:** user-uploaded Seed14 100GS `train(10).jsonl`, `aux(1).jsonl`, `eval(1).jsonl`, `updates(2).jsonl`, `adam_audit(1).jsonl`, original minimal source ZIP, and checkpoints `best(1).pt` (GS16), `latest(1).pt` (GS100). No test result has been selected after looking at reward. The new project has **not** passed a fresh 32/100GS convergence test.

## Observed failure: not a plot artifact

- Training 100GS: 50 D updates, 800 S updates, 897 auxiliary D80 physical calls. The on-line process used a 243-replica layout for much of GS65–100, frequently 0.45–0.55s E2E, while frozen validation diverged.
- GS16 held-out four-GS validation: mean Reward -6.0341, E2E .5711s, zero unfinished.
- GS40 held-out four-GS validation: Reward -26.4985, E2E 2.6315s, 68 unfinished.
- GS100 held-out four-GS validation: Reward -26.4613, E2E 2.6280s, 68 unfinished. Same external request trace hash at all checkpoints.
- GS40–100: initial BUILD proposes **all 896/896 stage-node entries**, original MILP keeps exactly **299**, 597 bits projected. Thereafter all three held-out periods KEEP the same matrix. Hence the flat validation curve is explained by *policy collapse + deterministic legal projection*, not fabricated statistics.
- Original capacity LP on the actual held-out matrices: GS16 ideal max CPU rho .435/.462/.464/.428 over periods 1–4; GS40/100 rho **1.106/1.172/1.225/1.377**. In the latter case individual stage CPU pressure stays below 1, establishing cross-stage shared-resource overcommit rather than a single-stage necessary overload.

## Identified training causes (distinguish evidence from inference)

1. **Wrong scale for the intended joint objective (confirmed in code).** `run(seed,arm='A')` actually constructs `Policy(base,'D')`. In `aux`, the conditional Refresh/Count/Node policy-gradient term is multiplied by `1.0` when `p.arm=='D'`, rather than by the detached macro `p_BUILD`. The macro gradient still uses `p_BUILD`. This was an experimental conditional BUILD-learning variant, **not** the gradient of the joint action objective. In uploaded GS100 auxiliary logs `p_BUILD ≈ 0.00003258`, while the conditional term still has weight 1. This mismatch can let count learn toward saturation even as the macro policy overwhelmingly chooses KEEP.
2. **Learning-rate mismatch (confirmed).** The authority configuration `configs/experiment_r0_joint.json` sets `r0_d_lr = 0.0003`; the minimal trainer overrode it with `Adam(lr=0.01)` — over **33 times larger**. The original update lacked rollback or per-head KL guard. This strongly amplifies all gradient/credit noise; it is not proof the optimizer alone caused every regression.
3. **Many-to-one MILP projection (confirmed).** Very different high-density proposals may map to the same RAM/SSD-legal deployment, making conditional action differentiation ineffective and holding/changing costs misleading if one reads only proposed replica counts. It is not a MILP implementation failure; no decoder/physical semantics are changed.
4. **Sparse, high-variance D80 stage credit and nonstationary joint S (known limits).** These are real features of the original formulation; this patch does not claim to solve all credit attribution issues or continuous D120 failures.

### Actual checkpoint-swap physical experiment

In two fixed Markov 4GS validation trajectories (8 commits per combination), loading actual historical checkpoints and swapping the D and S networks produced:

| D checkpoint | S checkpoint | mean Reward | request-weighted E2E | unfinished |
|---|---|---:|---:|---:|
| GS16 | GS16 | -5.2540 | .4979s | 0 |
| GS16 | GS100 | **-4.5732** | **.4272s** | 0 |
| GS100 | GS16 | -16.3759 | 1.7145s | 70 |

The fourth cross-combination `D100/S100` was not completed in the crossover script due to execution timeout, but the original frozen GS100 evaluation already records poor performance on the original seed 975101. This is a diagnostic, not a second independent seed. The result isolates deterioration mainly to D rather than ongoing training of S.

## Corrections made (no physical environment or action-space restrictions)

1. Correct conditional policy-gradient weight to `p_BUILD.detach()`; leave true conditional row credit and stage importance weights intact. This matches `J=(1-p_B)Q_KEEP+p_B E_{a|BUILD}[Q_BUILD(a)]` under a factorized macro / conditional distribution. The rare event of BUILD is **not** forced by a threshold.
2. Restore D learning rate from the authoritative config `x['r0_d_lr']` (currently **0.0003**).
3. Add a modest normalized entropy term for macro, refresh, binomial count, node distributions. This is regularization, **not** a count cap. All 1..32 counts and all legal stage/node choices retain probability support.
4. Add **PPO-free** bounded Adam update: exact KL for the macro categorical, refresh Bernoulli, count Binomial(31,p), and a first-node categorical proxy for the Plackett–Luce node order. KL is checked on current and previously observed **natural training states only**. On violation, restore parameters **and Adam state** and retry the **same gradient** with a reduced step. A failed update is skipped with audit; **physical actions are never filtered** and no validation Reward enters this decision. The node bound is an approximation for the full ordered-without-replacement action, not a full node-order KL proof.
5. Use 2 pre-frozen validation Markov demand seeds (975101, 975103), 4GS each, instead of one 4GS track. Both fixed before any new training; selection still uses real mean Reward, no post-hoc test-set selection. Note selection on eight GS is less noisy but not an unbiased complete-natural-demand estimator.
6. Save latest model plus optimizer, environment, RNG, Replay and training-only KL anchor observations in `latest.pt`; save only D/S weights in `best.pt`. A resume from the old algorithm checkpoint is **explicitly rejected** unless migrated/audited; start new training in a fresh directory. No historical weights are shipped.

## Verification status

**Passed:** source Python compilation; seven unit tests (including joint-gradient identity, exact Binomial KL, truthful optimizer rollback, checkpoint round-trip); valid action/physical replay of two main training GS using the user's original D80 scalar labels; D parameters changed, 16 S-DDQN updates, Replay populated, actor update KL under thresholds; GS0 physical frozen validation ran and produced only best/latest. A deliberately synthetic 50-step Count-pressure stress test reached saturation under old `.01` optimizer but retained mean Count p≈.286 under the corrected optimizer; this is **not** physical RL convergence evidence.

**Not passed / not claimed:** a fresh independent two-seed 32GS/100GS physical convergence run, a clean two-GS run in which all long D80 branches are freshly measured rather than using immutable existing scalar labels, all-seed/D120 reliability, or complete global optimality. An attempted fresh D80 two-GS run executed GS1 and was interrupted by physical-runtime timeout before the complete GS2 checkpoint. This limitation is retained, not hidden.

## Falsifiable acceptance criteria for the next full run

- Run from truly random D/S initialization; first 8GS smoke, 16/32/64/100 checkpoints at the fixed validation seeds (the program keeps only best/latest model files, but all numeric GS checkpoints stay in logs).
- Verify all D/S gradients finite; positive D update acceptance ratio; no unexpected sharp count-logit saturation/proposal 896 for many consecutive checkpoints; KL fields recorded. Zero reward/physical filtering.
- Verify both seeds show overall online and held-out Reward gains and E2E reduction vs GS0; compare frozen D/S swaps and S-only. Do **not** call this patch validated if both seeds reproduce GS40 degradation or if D updates are always rejected.
- Preserve real CPU/communication/MILP/Reward and D80/D120 definitions. Report any negative findings without modifying request seeds.
