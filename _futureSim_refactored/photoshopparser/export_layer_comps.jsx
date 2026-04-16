/*
 * export_layer_comps.jsx — run on an outer variant PSB (e.g. parade_timeline.psb
 * or parade_library.psb).
 *
 * For every layer comp in the active document (skipping names prefixed
 * "hidden_"), duplicates the doc, applies the comp, and saves a canvas-sized
 * TIFF to
 *   _data-refactored/_psds/exports/<variant>_<comp>.tiff
 * where <variant> is the active doc's name minus the .psb/.psd extension.
 *
 * TIFF is saved with layers=false (flat composite) + alphaChannels=true, so
 * transparency from the composite is preserved — no flatten/merge needed,
 * since Photoshop writes the composited pixels with their natural alpha
 * when layers are excluded.
 *
 * Leaves the source doc untouched.
 */
(function () {
    if (app.documents.length === 0) return "ERROR: no doc";
    var src = app.activeDocument;

    var variant = src.name.replace(/\.(psb|psd)$/i, "");

    var REPO_ROOT = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia";
    var OUT_DIR = new Folder(REPO_ROOT + "/_data-refactored/_psds/exports");
    if (!OUT_DIR.exists) OUT_DIR.create();

    var COMPS = [];
    for (var ci = 0; ci < src.layerComps.length; ci++) {
        var n = src.layerComps[ci].name;
        if (n.indexOf("hidden_") === 0) continue;
        COMPS.push(n);
    }
    var report = [];
    if (COMPS.length === 0) {
        report.push("NO COMPS — exporting current state as single TIFF");
        var dup = src.duplicate(variant + "_exp", false);
        app.activeDocument = dup;
        try {
            for (var li = 0; li < dup.layers.length; li++) {
                if (dup.layers[li].isBackgroundLayer) {
                    dup.layers[li].isBackgroundLayer = false;
                    break;
                }
            }
            var tifTmp = new File(OUT_DIR.fsName + "/" + variant + ".tif");
            var tifOpts = new TiffSaveOptions();
            tifOpts.imageCompression = TIFFEncoding.TIFFLZW;
            tifOpts.layers = false;
            tifOpts.alphaChannels = true;
            tifOpts.transparency = true;
            tifOpts.embedColorProfile = true;
            dup.saveAs(tifTmp, tifOpts, true, Extension.LOWERCASE);
            var tifFinal = new File(OUT_DIR.fsName + "/" + variant + ".tiff");
            if (tifFinal.exists) tifFinal.remove();
            tifTmp.rename(variant + ".tiff");
            report.push("WROTE: " + tifFinal.fsName);
        } catch (e) {
            report.push("ERROR: " + e);
        }
        dup.close(SaveOptions.DONOTSAVECHANGES);
        app.activeDocument = src;
        return report.join("\n");
    }

    report.push("COMPS: " + COMPS.join(", "));

    for (var c = 0; c < COMPS.length; c++) {
        var compName = COMPS[c];
        var comp = findComp(src, compName);
        if (!comp) {
            report.push("MISSING COMP: " + compName);
            continue;
        }

        var dup = src.duplicate(variant + "_exp_" + compName, false);
        app.activeDocument = dup;
        try {
            var dupComp = findComp(dup, compName);
            if (!dupComp) {
                report.push("MISSING COMP in dup: " + compName);
                dup.close(SaveOptions.DONOTSAVECHANGES);
                continue;
            }
            dupComp.apply();
            // If the doc has a locked Background layer, convert it to a
            // normal layer so the composite retains transparency. Without
            // this, Photoshop bakes the background white into the TIFF.
            for (var li = 0; li < dup.layers.length; li++) {
                if (dup.layers[li].isBackgroundLayer) {
                    dup.layers[li].isBackgroundLayer = false;
                    break;
                }
            }

            var stem = variant + "_" + compName;
            var tifTmp = new File(OUT_DIR.fsName + "/" + stem + ".tif");
            var tifOpts = new TiffSaveOptions();
            tifOpts.imageCompression = TIFFEncoding.TIFFLZW;
            tifOpts.layers = false;
            tifOpts.alphaChannels = true;
            tifOpts.transparency = true;
            tifOpts.embedColorProfile = true;
            dup.saveAs(tifTmp, tifOpts, true, Extension.LOWERCASE);
            // Photoshop normalizes the TIFF extension to .tif; rename to .tiff.
            var tifFinal = new File(OUT_DIR.fsName + "/" + stem + ".tiff");
            if (tifFinal.exists) tifFinal.remove();
            tifTmp.rename(stem + ".tiff");
            report.push("WROTE: " + tifFinal.fsName);
        } catch (e) {
            report.push("ERROR " + compName + ": " + e);
        }
        dup.close(SaveOptions.DONOTSAVECHANGES);
        app.activeDocument = src;
    }

    return report.join("\n");

    function findComp(doc, n) {
        var C = doc.layerComps;
        for (var i = 0; i < C.length; i++) if (C[i].name === n) return C[i];
        return null;
    }
})();
