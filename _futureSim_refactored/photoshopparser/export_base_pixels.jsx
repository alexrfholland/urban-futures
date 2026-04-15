/*
 * export_base_pixels.jsx — run on base_parade_timeline.psb.
 *
 * Exports the 3 pixel layers (base_rgb, base_rgb copy, base ambient
 * occlusion) as canvas-sized PNGs into
 *   _data-refactored/_psds/linked_pngs/template_base/<layer>.png
 *
 * Approach: duplicate the doc, in the dup hide everything except the
 * target layer + disable adjustment layers, flatten-for-export via
 * saveAs PNG, close dup. Keeps the source PSB untouched.
 */
(function () {
    if (app.documents.length === 0) return "ERROR: no doc";
    var src = app.activeDocument;
    if (!/^base_parade_timeline\.psb$/i.test(src.name)) {
        return "ERROR: active doc is '" + src.name
             + "', expected base_parade_timeline.psb";
    }

    var REPO_ROOT = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia";
    var OUT_DIR = new Folder(REPO_ROOT
        + "/_data-refactored/_psds/linked_pngs/template_base");
    if (!OUT_DIR.exists) OUT_DIR.create();

    var TARGETS = ["base_rgb", "base_rgb copy", "base ambient occlusion"];
    var report = [];

    for (var t = 0; t < TARGETS.length; t++) {
        var name = TARGETS[t];
        var dup = src.duplicate(src.name + "_exp_" + t, false);
        app.activeDocument = dup;
        try {
            // Hide every top-level layer, then show only the target.
            for (var i = 0; i < dup.layers.length; i++) {
                dup.layers[i].visible = false;
            }
            var target = findChild(dup, name);
            if (!target) {
                report.push("MISSING: " + name);
                dup.close(SaveOptions.DONOTSAVECHANGES);
                continue;
            }
            target.visible = true;
            // Flatten to render just this layer's pixels on the canvas.
            dup.flatten();
            var out = new File(OUT_DIR.fsName + "/" + name + ".png");
            var opts = new PNGSaveOptions();
            opts.compression = 6;
            opts.interlaced = false;
            dup.saveAs(out, opts, true, Extension.LOWERCASE);
            report.push("WROTE: " + out.fsName);
        } catch (e) {
            report.push("ERROR " + name + ": " + e);
        }
        dup.close(SaveOptions.DONOTSAVECHANGES);
        app.activeDocument = src;
    }

    return report.join("\n");

    function findChild(parent, n) {
        var L = parent.layers;
        for (var i = 0; i < L.length; i++) if (L[i].name === n) return L[i];
        return null;
    }
})();
