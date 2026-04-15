// Propagate a smart object (linked PNG) across one or more open outer-variant PSBs.
// Edit the CONFIG block below to describe: source SO name + layer path (for placement),
// which parent groups to populate, and per-target linked_pngs site dir.
// The script is idempotent: reruns skip SOs that are already linked to the right file.
//
// Placement rule: a new group is positioned so it sits AFTER the 'anchor' group
// (e.g. 'arboreal'). Do NOT default to top-of-parent — mirror the source PSB's order.
// See §9 of PSD_TEMPLATE_CONTRACT.md for the mistake that motivated this rule.
//
// PS quirk: layer.move(anchor, PLACEAFTER) pops the layer OUT of its parent when
// anchor is the last child. Workaround here: create the group via
// doc.layerSets.add() -> move BEFORE anchor (lands inside parent), then
// anchor.move(newGroup, PLACEBEFORE) to swap into [anchor, newGroup, ...].

// ---------- CONFIG ----------
var CONFIG = {
    so_name: "base_outlines",              // layer name inside target group
    group_name: "outlines base",           // subgroup name (created under each parent)
    png_file: "base_outlines.png",         // filename looked up under <png_dir>/<parent>/<group>/
    anchor_name: "arboreal",               // the new group is placed AFTER this sibling
    parents: ["Trending", "Positive"],     // parent groups to populate (skip if missing)
    targets: [
        // { doc: "<doc-name.psb>", png_dir: "/abs/path/to/linked_pngs/<site>" }
        {doc: "parade_timeline.psb",               png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/parade"},
        {doc: "uni_timeline.psb",                  png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/uni_timeline"},
        {doc: "city_timeline.psb",                 png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/city_timeline"},
        {doc: "parade_single-state_yr180.psb",     png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/parade_single-state_yr180"},
        {doc: "city_single-state_yr180.psb",       png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/city_single-state_yr180"},
        {doc: "parade_baseline__hero-camera.psb",  png_dir: "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/_psds/linked_pngs/parade_baseline__hero-camera"}
    ]
};
// ---------- end CONFIG ----------

function findDoc(name) {
    for (var i = 0; i < app.documents.length; i++) {
        if (app.documents[i].name == name) return app.documents[i];
    }
    return null;
}
function findChildSet(parent, name) {
    if (!parent) return null;
    for (var i = 0; i < parent.layers.length; i++) {
        var L = parent.layers[i];
        if (L.typename == "LayerSet" && L.name == name) return L;
    }
    return null;
}
function findChildLayer(parent, name) {
    if (!parent) return null;
    for (var i = 0; i < parent.layers.length; i++) {
        if (parent.layers[i].name == name) return parent.layers[i];
    }
    return null;
}
function isSmartObject(layer) {
    try {
        var ref = new ActionReference();
        ref.putProperty(stringIDToTypeID("property"), stringIDToTypeID("smartObject"));
        ref.putIdentifier(stringIDToTypeID("layer"), layer.id);
        executeActionGet(ref);
        return true;
    } catch(e) { return false; }
}
function soIsLinked(doc, layer) {
    try {
        doc.activeLayer = layer;
        var ref = new ActionReference();
        ref.putIdentifier(stringIDToTypeID("layer"), layer.id);
        var d = executeActionGet(ref);
        var so = d.getObjectValue(stringIDToTypeID("smartObject"));
        return so.hasKey(stringIDToTypeID("linked")) && so.getBoolean(stringIDToTypeID("linked"));
    } catch(e) { return false; }
}
function placeLinked(doc, groupLayer, pngPath, soName) {
    doc.activeLayer = groupLayer;
    var desc = new ActionDescriptor();
    desc.putPath(charIDToTypeID("null"), new File(pngPath));
    desc.putBoolean(stringIDToTypeID("linked"), true);
    executeAction(stringIDToTypeID("placeEvent"), desc, DialogModes.NO);
    var placed = doc.activeLayer;
    placed.name = soName;
    if (placed.parent !== groupLayer) {
        placed.move(groupLayer, ElementPlacement.PLACEATBEGINNING);
    }
    return placed;
}
// Ensure `group_name` exists inside `parent`, positioned directly after `anchor`.
// Uses the swap workaround for the PS PLACEAFTER-last-child quirk.
function ensureGroupAfterAnchor(doc, parent, anchor, groupName) {
    var g = findChildSet(parent, groupName);
    if (!g) {
        var newG = doc.layerSets.add();
        newG.name = groupName;
        newG.move(anchor, ElementPlacement.PLACEBEFORE);
        g = newG;
    }
    var idxA = -1, idxG = -1;
    for (var i = 0; i < parent.layers.length; i++) {
        if (parent.layers[i] === anchor) idxA = i;
        if (parent.layers[i] === g) idxG = i;
    }
    // In PS stack, lower index = higher in the panel (visually on top). We want
    // anchor above group: idxA < idxG. If order is reversed, swap.
    if (idxA > idxG) {
        anchor.move(g, ElementPlacement.PLACEBEFORE);
    }
    return g;
}

var report = [];
for (var t = 0; t < CONFIG.targets.length; t++) {
    var tgt = CONFIG.targets[t];
    var doc = findDoc(tgt.doc);
    if (!doc) { report.push(tgt.doc + ": NOT OPEN"); continue; }
    app.activeDocument = doc;
    for (var p = 0; p < CONFIG.parents.length; p++) {
        var parentName = CONFIG.parents[p];
        var parent = findChildSet(doc, parentName);
        if (!parent) { report.push(tgt.doc + "/" + parentName + ": MISSING parent"); continue; }
        var anchor = findChildSet(parent, CONFIG.anchor_name);
        if (!anchor) { report.push(tgt.doc + "/" + parentName + ": NO " + CONFIG.anchor_name + ", skip"); continue; }

        var pngPath = tgt.png_dir + "/" + parentName + "/" + CONFIG.group_name + "/" + CONFIG.png_file;
        var g = ensureGroupAfterAnchor(doc, parent, anchor, CONFIG.group_name);

        var existing = findChildLayer(g, CONFIG.so_name);
        if (existing) {
            var keep = isSmartObject(existing) && soIsLinked(doc, existing);
            if (!keep) {
                existing.remove();
                report.push(tgt.doc + "/" + parentName + "/" + CONFIG.group_name + ": removed stale " + CONFIG.so_name);
                existing = null;
            }
        }
        if (!existing) {
            placeLinked(doc, g, pngPath, CONFIG.so_name);
            report.push(tgt.doc + "/" + parentName + "/" + CONFIG.group_name + ": LINKED " + pngPath);
        } else {
            report.push(tgt.doc + "/" + parentName + "/" + CONFIG.group_name + ": already linked, skip");
        }
    }
    doc.save();
    report.push(tgt.doc + ": saved");
}
report.join("\n");
