/*
 * generate_base_template.jsx — builds _psds/psd-live/base_template.psd
 * from scratch, matching parade-base.psd's structure but using 4.12
 * parade_timeline's 8K base PNGs as linked smart objects so transforms
 * are identity at 7680x4320 (no baked trim bbox to fight later).
 *
 * Layer stack (top -> bottom), matching parade-base.psd:
 *   base_depth_windowed_balanced_dense   (linked SO, NORMAL, visible)
 *   base_depth_windowed_internal_refined (linked SO, NORMAL, hidden)
 *   Hue/Saturation 1              (master=(0,-73,0), colorize=false)
 *   base_rgb                      (linked SO, COLOR,  base_rgb.png)
 *   existing_condition_ao_full   (linked SO, NORMAL, base_white_render.png)
 *
 * Canvas: 7680 x 4320 RGB 8-bit.
 */

(function () {
    var REPO_ROOT = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia";
    var SRC_DIR = REPO_ROOT
        + "/_data-refactored/compositor/outputs/4.12/parade_timeline/base__20260414_214548";
    var OUT = new File(REPO_ROOT + "/_data-refactored/_psds/psd-live/base_template.psd");

    // Placement order: bottom layer first so stack ends up correct top-down.
    // parade-base top->bottom: depth_dense, depth_internal, huesat, rgb, ao,
    // so placement (bottom first) is the reverse.
    var STACK = [
        { file: "base_white_render.png",
          name: "existing_condition_ao_full",
          blend: BlendMode.NORMAL, visible: true },
        { file: "base_rgb.png",
          name: "base_rgb",
          blend: BlendMode.COLORBLEND, visible: true },
        { kind: "huesat" },
        { file: "base_depth_windowed_internal_refined.png",
          name: "base_depth_windowed_internal_refined",
          blend: BlendMode.NORMAL, visible: false },
        { file: "base_depth_windowed_balanced_dense.png",
          name: "base_depth_windowed_balanced_dense",
          blend: BlendMode.NORMAL, visible: true },
    ];

    // Sanity-check inputs
    for (var s = 0; s < STACK.length; s++) {
        if (STACK[s].file) {
            var f = new File(SRC_DIR + "/" + STACK[s].file);
            if (!f.exists) return "ERROR: missing input " + f.fsName;
        }
    }

    // Clean start: close anything open so we don't accidentally place into
    // someone else's doc.
    while (app.documents.length > 0) {
        app.documents[0].close(SaveOptions.DONOTSAVECHANGES);
    }

    var savedUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    var doc = app.documents.add(
        new UnitValue(7680, "px"),
        new UnitValue(4320, "px"),
        72, "base_template",
        NewDocumentMode.RGB,
        DocumentFill.TRANSPARENT,
        1.0, BitsPerChannelType.EIGHT);

    var report = [];

    for (var i = 0; i < STACK.length; i++) {
        var entry = STACK[i];
        if (entry.kind === "huesat") {
            addHueSaturationAdjustment(0, -73, 0);
            report.push("added HueSat (master=-73)");
            continue;
        }
        var pngPath = new File(SRC_DIR + "/" + entry.file);
        placeLinkedFromFile(pngPath);
        var layer = doc.activeLayer;
        layer.name = entry.name;
        layer.blendMode = entry.blend;
        layer.visible = entry.visible;
        report.push("placed " + entry.name
                  + " (" + layer.bounds[0].as("px") + ","
                  + layer.bounds[1].as("px") + " -> "
                  + layer.bounds[2].as("px") + ","
                  + layer.bounds[3].as("px") + ")");
    }

    // Remove the transparent starting layer (Photoshop adds one when doc is
    // created). It's named "Layer 1" or similar and sits at the bottom.
    for (var j = doc.artLayers.length - 1; j >= 0; j--) {
        var L = doc.artLayers[j];
        var recognised = false;
        for (var k = 0; k < STACK.length; k++) {
            if (STACK[k].name && L.name === STACK[k].name) { recognised = true; break; }
        }
        if (!recognised && L.kind === LayerKind.NORMAL && L.name.indexOf("Layer") === 0) {
            L.remove();
            report.push("removed starter layer: " + L.name);
        }
    }

    var psdOpts = new PhotoshopSaveOptions();
    psdOpts.embedColorProfile = true;
    psdOpts.alphaChannels = true;
    psdOpts.layers = true;
    psdOpts.maximizeCompatibility = true;
    doc.saveAs(OUT, psdOpts, true, Extension.LOWERCASE);
    report.push("SAVED " + OUT.fsName);

    app.preferences.rulerUnits = savedUnits;
    return report.join("\n");

    // ---- helpers -------------------------------------------------------

    function placeLinkedFromFile(file) {
        var desc = new ActionDescriptor();
        desc.putPath(charIDToTypeID("null"), file);
        desc.putBoolean(stringIDToTypeID("linked"), true);
        executeAction(charIDToTypeID("Plc "), desc, DialogModes.NO);
    }

    function addHueSaturationAdjustment(hue, sat, light) {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putClass(stringIDToTypeID("adjustmentLayer"));
        desc.putReference(charIDToTypeID("null"), ref);

        var using = new ActionDescriptor();
        var adj = new ActionDescriptor();
        adj.putBoolean(stringIDToTypeID("colorize"), false);
        using.putObject(charIDToTypeID("Type"), stringIDToTypeID("hueSaturation"), adj);
        desc.putObject(charIDToTypeID("Usng"), stringIDToTypeID("adjustmentLayer"), using);
        executeAction(charIDToTypeID("Mk  "), desc, DialogModes.NO);

        // Now set the hue/sat/lightness values on the freshly-created layer
        var set = new ActionDescriptor();
        var setRef = new ActionReference();
        setRef.putEnumerated(charIDToTypeID("AdjL"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        set.putReference(charIDToTypeID("null"), setRef);

        var adjParams = new ActionDescriptor();
        adjParams.putBoolean(stringIDToTypeID("colorize"), false);
        var adjustment = new ActionList();
        var entry = new ActionDescriptor();
        entry.putInteger(charIDToTypeID("H   "), hue);
        entry.putInteger(charIDToTypeID("Strt"), sat);
        entry.putInteger(charIDToTypeID("Lght"), light);
        adjustment.putObject(stringIDToTypeID("hueSatAdjustmentV2"), entry);
        adjParams.putList(stringIDToTypeID("adjustment"), adjustment);
        set.putObject(charIDToTypeID("T   "), stringIDToTypeID("hueSaturation"), adjParams);
        executeAction(charIDToTypeID("setd"), set, DialogModes.NO);
    }
})();
