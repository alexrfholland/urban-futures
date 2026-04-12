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


# -- Bioenvelope PLY classification colours ---------------------------------
# RGB tuples for the proposal-based bioenvelope classification.
# Used by export_proposal_envelopes and any downstream visualisation.
BIOENVELOPE_PLY_COLORS: dict[str, tuple[int, int, int]] = {
    RECRUIT_FULL:            (142, 216, 200),  # mint    #8ED8C8
    RECRUIT_PARTIAL:         (240, 220, 144),  # butter  #F0DC90
    COLONISE_FULL_ENVELOPE:  (184, 232, 108),  # lime    #B8E86C
    COLONISE_PARTIAL_ENVELOPE: (208, 160,  64),  # ochre   #D0A040
    COLONISE_FULL_GROUND:    (142, 216, 200),  # mint    #8ED8C8
    DECAY_FULL:              (240, 220, 144),  # butter  #F0DC90
    "deploy-any":            (220, 192, 144),  # sand    #DCC090
    "buffer-feature+depaved": (220, 120, 160),  # pink    #DC78A0
    "none":                  (255, 255, 255),  # white
}

# -- Bioenvelope PLY int mapping --------------------------------------------
# Single-channel integer encoding for ``intervention_bioenvelope_ply-int``.
# 0 = none.  Higher values follow the priority order.
BIOENVELOPE_PLY_INT: dict[str, int] = {
    "none":                    0,
    "deploy-any":              1,
    DECAY_FULL:                2,
    COLONISE_FULL_GROUND:      3,
    COLONISE_PARTIAL_ENVELOPE: 4,
    COLONISE_FULL_ENVELOPE:    5,
    RECRUIT_PARTIAL:           6,
    RECRUIT_FULL:              7,
    "buffer-feature+depaved":  8,
}
