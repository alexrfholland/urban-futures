"""
Intervention label constants for the v4 proposal system.

Import these everywhere instead of hardcoding intervention strings.
Change values here to relabel interventions across the entire pipeline.
"""

# -- Decay -------------------------------------------------------------------
DECAY_FULL = "buffer-feature"
DECAY_PARTIAL = "brace-feature"

# -- Recruit -----------------------------------------------------------------
RECRUIT_FULL = "rewild-larger-patch"
RECRUIT_PARTIAL = "rewild-smaller-patch"

# -- Deploy ------------------------------------------------------------------
DEPLOY_FULL_POLE = "adapt-utility-pole"
DEPLOY_FULL_LOG = "translocate-log"
DEPLOY_FULL_UPGRADE = "upgrade-feature"
DEPLOY_GROUND_DEADWOOD = "translocate-deadwood"

# -- Colonise ----------------------------------------------------------------
COLONISE_FULL_GROUND = "larger-patches-rewild"
COLONISE_FULL_ENVELOPE = "enrich-envelope"
COLONISE_PARTIAL_ENVELOPE = "roughen-envelope"

# -- Release Control ---------------------------------------------------------
RELEASECONTROL_FULL = "eliminate-canopy-pruning"
RELEASECONTROL_PARTIAL = "reduce-canopy-pruning"
