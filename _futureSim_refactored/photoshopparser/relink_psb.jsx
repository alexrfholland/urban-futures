/*
 * relink_psb.jsx — ExtendScript for Photoshop.
 *
 * On the active PSB, runs three idempotent setup passes:
 *   1. Relinks each of the THREE "Base" smart objects to
 *      _data-refactored/_psds/psd-live/base_<variant>.psb. The outer
 *      template ships with all three pointing at base_template.psd; this
 *      pass swaps them to the per-variant base PSB. Uses
 *      placedLayerRelinkToFile, which preserves the layer's clipping-mask
 *      flag and stack position (so the proposal-release-control-* clip
 *      relationships survive).
 *   2. For non-timeline variants (single-state, baseline, ...), prunes the
 *      per-branch "Hue/Saturation - medium(n)" adjustment layers under
 *      Trending/...sizes and Positive/...sizes (timeline-only).
 *   3. Repoints each populated smart object at its matching PNG in
 *      _data-refactored/_psds/linked_pngs/<site>/. Converts embedded
 *      smart objects to linked in the process.
 *
 * Source of truth for (3): the linked_pngs/<site>/ folder itself. Every
 * .png's path relative to that folder is the PSB layer path it belongs to.
 *
 * The <site> is derived from the active document's filename:
 *   parade_timeline.psb               -> parade
 *   parade_timeline test.psb          -> parade
 *   parade_single-state_yr180.psb     -> parade_single-state_yr180
 */

(function () {
    if (app.documents.length === 0) {
        return "ERROR: no document open";
    }
    var doc = app.activeDocument;

    var site = deriveSite(doc.name);
    if (!site) {
        return "ERROR: can't derive site from doc name: " + doc.name;
    }

    var REPO_ROOT = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia";
    var PSD_LIVE = REPO_ROOT + "/_data-refactored/_psds/psd-live";
    // Base PSB mirrors the outer doc name: <variant>.psb -> base_<variant>.psb
    var variantStem = doc.name.replace(/\.psb$/i, "").replace(/\.psd$/i, "");
    var BASE_PSB = new File(PSD_LIVE + "/base_" + variantStem + ".psb");
    var LINKED_ROOT = REPO_ROOT + "/_data-refactored/_psds/linked_pngs/" + site;

    var report = [];

    // --- Pass 1: relink the THREE Base smart objects ----------------------
    // Outer template ships with these pointing at base_template.psd; we
    // swap each one to the per-variant base PSB. relinkToFile preserves
    // clipping-mask state and stack position.
    var BASE_PATHS = [
        ["Trending", "arboreal", "trending",
         "proposal-release-control-rejected", "Base"],
        ["Trending", "arboreal", "trending",
         "proposal-release-control-reduce-canopy-pruning", "Base"],
        ["BASE WORLD", "Base"]
    ];
    if (!BASE_PSB.exists) {
        report.push("base: ERROR target missing " + BASE_PSB.fsName);
    } else {
        for (var bp = 0; bp < BASE_PATHS.length; bp++) {
            var bpath = BASE_PATHS[bp];
            var baseLayer = navigate(doc, bpath);
            var label = "base [" + bpath.join("/") + "]";
            if (!baseLayer) {
                report.push(label + ": (missing — skipping)");
                continue;
            }
            try {
                if (isLinkedToFile(baseLayer, BASE_PSB)) {
                    report.push(label + ": already linked");
                } else {
                    doc.activeLayer = baseLayer;
                    relinkToFile(BASE_PSB);
                    report.push(label + ": relinked to " + BASE_PSB.fsName);
                }
            } catch (e) {
                report.push(label + ": ERROR " + e);
            }
        }
    }

    // --- Pass 2: prune timeline-only hue adjustments ---------------------
    var isTimeline = /timeline/i.test(doc.name);
    if (!isTimeline) {
        var HUE_TARGETS = [
            ["Trending", "arboreal", "trending", "sizes", "Hue/Saturation - medium"],
            ["Positive", "arboreal", "positive", "sizes", "Hue/Saturation - mediun"],
        ];
        for (var h = 0; h < HUE_TARGETS.length; h++) {
            var hpath = HUE_TARGETS[h];
            var hlayer = navigate(doc, hpath);
            if (hlayer) {
                hlayer.remove();
                report.push("hue pruned: " + hpath.join("/"));
            } else {
                report.push("hue: (already absent) " + hpath.join("/"));
            }
        }
    } else {
        report.push("hue: (timeline doc — keeping adjustments)");
    }

    // --- Pass 3: relink PNG smart objects --------------------------------
    var root = new Folder(LINKED_ROOT);
    if (!root.exists) {
        doc.save();
        report.push("SAVED (pngs skipped — linked_pngs root missing: " + LINKED_ROOT + ")");
        return report.join("\n");
    }

    var pngs = [];
    collectPngs(root, LINKED_ROOT, pngs);

    var stats = { relinked: 0, converted_to_smart_object: 0,
                  already_linked: 0, missing_layer: 0, error: 0 };

    for (var i = 0; i < pngs.length; i++) {
        var entry = pngs[i];
        var layer = navigate(doc, entry.path);
        if (!layer) {
            stats.missing_layer++;
            report.push("MISSING LAYER: " + entry.path.join("/"));
            continue;
        }
        try {
            if (isLinkedToFile(layer, entry.file)) {
                stats.already_linked++;
                continue;
            }
            doc.activeLayer = layer;
            if (!isSmartObject(layer)) {
                convertToSmartObject();
                stats.converted_to_smart_object++;
            }
            relinkToFile(entry.file);
            stats.relinked++;
            report.push("RELINKED: " + entry.path.join("/"));
        } catch (e) {
            stats.error++;
            report.push("ERROR: " + entry.path.join("/") + " -> " + e);
        }
    }

    doc.save();

    report.push("");
    report.push("=== Summary ===");
    for (var k in stats) {
        report.push("  " + k + ": " + stats[k]);
    }
    report.push("SAVED");
    return report.join("\n");

    // --- helpers ----------------------------------------------------------

    function deriveSite(docName) {
        var base = docName.replace(/\.psb$/i, "").replace(/ test$/i, "");
        if (base === "parade_timeline") return "parade";
        return base;
    }

    function collectPngs(folder, rootPath, out) {
        var entries = folder.getFiles();
        for (var j = 0; j < entries.length; j++) {
            var f = entries[j];
            if (f instanceof Folder) {
                collectPngs(f, rootPath, out);
            } else if (f instanceof File && /\.png$/i.test(f.name)) {
                var rel = f.fsName.substring(rootPath.length + 1).replace(/\\/g, "/");
                var parts = rel.replace(/\.png$/i, "").split("/");
                out.push({ path: parts, file: f });
            }
        }
    }

    function navigate(rootLayer, segments) {
        var current = rootLayer;
        for (var j = 0; j < segments.length; j++) {
            current = findChild(current, segments[j]);
            if (!current) return null;
        }
        return current;
    }

    function findChild(parent, name) {
        var layers = parent.layers;
        for (var k = 0; k < layers.length; k++) {
            if (layers[k].name === name) return layers[k];
        }
        return null;
    }

    function isLinkedToFile(layer, file) {
        try {
            var ref = new ActionReference();
            ref.putIdentifier(charIDToTypeID("Lyr "), layer.id);
            var desc = executeActionGet(ref);
            var smartId = stringIDToTypeID("smartObject");
            if (!desc.hasKey(smartId)) return false;
            var so = desc.getObjectValue(smartId);
            var linkedId = stringIDToTypeID("linked");
            if (!so.hasKey(linkedId) || !so.getBoolean(linkedId)) return false;
            var fileRefId = stringIDToTypeID("fileReference");
            if (!so.hasKey(fileRefId)) return false;
            var linkedPath = so.getString(fileRefId);
            return linkedPath === file.fsName;
        } catch (e) {
            return false;
        }
    }

    function relinkToFile(file) {
        var desc = new ActionDescriptor();
        desc.putPath(charIDToTypeID("null"), file);
        executeAction(stringIDToTypeID("placedLayerRelinkToFile"), desc, DialogModes.NO);
    }

    function isSmartObject(layer) {
        try {
            var ref = new ActionReference();
            ref.putIdentifier(charIDToTypeID("Lyr "), layer.id);
            var desc = executeActionGet(ref);
            return desc.hasKey(stringIDToTypeID("smartObject"));
        } catch (e) {
            return false;
        }
    }

    function convertToSmartObject() {
        executeAction(stringIDToTypeID("newPlacedLayer"), undefined, DialogModes.NO);
    }
})();
