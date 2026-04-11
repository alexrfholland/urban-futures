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


# -- Intervention support level ---------------------------------------------
# Maps each intervention to 'full' or 'partial'. Used by the proposal stream
# graph plotter to drive opacity (partials rendered at 50% opacity).
INTERVENTION_SUPPORT = {
    DECAY_FULL: "full",
    DECAY_PARTIAL: "partial",
    RELEASECONTROL_FULL: "full",
    RELEASECONTROL_PARTIAL: "partial",
    RECRUIT_FULL: "full",
    RECRUIT_PARTIAL: "partial",
    COLONISE_FULL_GROUND: "full",
    COLONISE_FULL_ENVELOPE: "full",
    COLONISE_PARTIAL_ENVELOPE: "partial",
    DEPLOY_FULL_POLE: "full",
    DEPLOY_FULL_LOG: "full",
    DEPLOY_FULL_UPGRADE: "full",
    DEPLOY_GROUND_DEADWOOD: "full",
}
