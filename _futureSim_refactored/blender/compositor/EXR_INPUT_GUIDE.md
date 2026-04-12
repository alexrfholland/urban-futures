Compositor-side lookup for bV2 EXR view layers; authoritative contract is [bV2_exr_aov_contract.md](../blenderv2/bV2_exr_aov_contract.md).

# EXRs

1. `existing_condition_positive`
2. `existing_condition_trending`
3. `positive_state`
4. `positive_priority_state`
5. `trending_state`
6. `bioenvelope_positive`
7. `bioenvelope_trending`

# Masks

Some compositor blends use an arboreal mask derived from `IndexOB == 3`.

# Per-Blend Notes

**proposal_and_interventions** — runs on 3-7. Does not use the arboreal mask.
!!