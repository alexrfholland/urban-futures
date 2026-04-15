/*
 * restructure_psb.jsx — ExtendScript for Photoshop.
 *
 * Converts specific groups from "group-level raster mask" to "clipping mask
 * driven by linked smart object inside the group". Idempotent.
 *
 * Targets (hardcoded for Trending pass):
 *   Trending/arboreal/trending/proposal-release-control-rejected
 *   Trending/arboreal/trending/proposal-release-control-reduce-canopy-pruning
 *
 * Runs on the active document. Expects the TEST PSB to be open.
 */

(function () {
    if (app.documents.length === 0) {
        return "ERROR: no document open";
    }
    var doc = app.activeDocument;
    if (!/parade_timeline test\.psb/i.test(doc.name)) {
        return "ERROR: wrong doc: " + doc.name;
    }

    var TARGETS = [
        ["Trending", "arboreal", "trending", "proposal-release-control-rejected"],
        ["Trending", "arboreal", "trending", "proposal-release-control-reduce-canopy-pruning"],
    ];

    var report = [];

    for (var i = 0; i < TARGETS.length; i++) {
        var path = TARGETS[i];
        var groupName = path[path.length - 1];
        var group = navigate(doc, path);
        if (!group) {
            report.push("MISSING: " + path.join("/"));
            continue;
        }

        doc.activeLayer = group;

        if (hasUserMask(group)) {
            deleteLayerMask();
            report.push(groupName + ": removed group mask");
        } else {
            report.push(groupName + ": (no mask to remove)");
        }

        var pngLayer = findChild(group, groupName);
        if (pngLayer) {
            if (!pngLayer.visible) {
                pngLayer.visible = true;
                report.push(groupName + ": set PNG layer visible");
            }
        } else {
            report.push(groupName + ": MISSING expected PNG layer");
            continue;
        }

        var baseLayer = findChild(group, "Base");
        if (baseLayer) {
            if (!baseLayer.grouped) {
                doc.activeLayer = baseLayer;
                baseLayer.grouped = true;
                report.push(groupName + ": clipped Base -> " + groupName);
            } else {
                report.push(groupName + ": (Base already clipped)");
            }
        } else {
            report.push(groupName + ": MISSING Base layer");
        }
    }

    doc.save();
    report.push("SAVED");

    return report.join("\n");

    // --- helpers ----------------------------------------------------------

    function navigate(root, segments) {
        var current = root;
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

    function hasUserMask(layer) {
        var ref = new ActionReference();
        ref.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        var desc = executeActionGet(ref);
        return desc.hasKey(stringIDToTypeID("hasUserMask")) && desc.getBoolean(stringIDToTypeID("hasUserMask"));
    }

    function deleteLayerMask() {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("Msk "));
        desc.putReference(charIDToTypeID("null"), ref);
        executeAction(charIDToTypeID("Dlt "), desc, DialogModes.NO);
    }
})();
