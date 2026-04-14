/*
  Photoshop ExtendScript
  Duplicates the active document, flattens it, optionally resizes it for
  InDesign placement, and saves a lossless TIFF to a sibling export folder.

  Usage:
  1. Open a layered PSD/PSB master in Photoshop.
  2. Edit CONFIG below.
  3. Run via File > Scripts > Browse...

  If you want a native Photoshop Action, record a one-step action that runs
  this script. That is more maintainable than a hand-built binary .atn file.
*/

var CONFIG = {
  // "FULL" keeps the document pixel dimensions.
  // "PLACE" resizes to the target print dimensions below.
  mode: "PLACE",

  // Only used when mode === "PLACE"
  widthInches: 7.56,
  heightInches: 10.31,
  ppi: 300,

  // Export destination:
  // - "SIBLING_FOLDER" creates/uses "<source folder>/Links_Flattened"
  // - "CUSTOM_FOLDER" uses absoluteOutputFolder
  outputMode: "SIBLING_FOLDER",
  siblingFolderName: "Links_Flattened",
  absoluteOutputFolder: "/Users/alexholland/Desktop/Links_Flattened",

  suffix: "_id-flat",
  tiffCompression: TIFFEncoding.TIFFZIP,
  embedColorProfile: true,
  overwriteExisting: true
};

function fail(message) {
  alert(message);
  throw new Error(message);
}

function ensureOpenDocument() {
  if (app.documents.length === 0) {
    fail("Open a PSD/PSB document first.");
  }
}

function ensureSavedDocument(doc) {
  if (!doc.saved || !doc.fullName) {
    fail("Save the source document first so the export path can be resolved.");
  }
}

function getOutputFolder(sourceFile) {
  if (CONFIG.outputMode === "CUSTOM_FOLDER") {
    var customFolder = new Folder(CONFIG.absoluteOutputFolder);
    if (!customFolder.exists && !customFolder.create()) {
      fail("Could not create output folder:\n" + customFolder.fsName);
    }
    return customFolder;
  }

  var siblingFolder = new Folder(sourceFile.parent.fsName + "/" + CONFIG.siblingFolderName);
  if (!siblingFolder.exists && !siblingFolder.create()) {
    fail("Could not create output folder:\n" + siblingFolder.fsName);
  }
  return siblingFolder;
}

function buildOutputFile(sourceFile, folder) {
  var baseName = sourceFile.name.replace(/\.[^\.]+$/, "");
  return new File(folder.fsName + "/" + baseName + CONFIG.suffix + ".tif");
}

function resizeForPlacement(doc) {
  if (CONFIG.mode !== "PLACE") {
    return;
  }

  var widthPx = Math.round(CONFIG.widthInches * CONFIG.ppi);
  var heightPx = Math.round(CONFIG.heightInches * CONFIG.ppi);

  doc.resizeImage(
    UnitValue(widthPx, "px"),
    UnitValue(heightPx, "px"),
    CONFIG.ppi,
    ResampleMethod.BICUBICSHARPER
  );
}

function saveAsTiff(doc, outputFile) {
  if (outputFile.exists && !CONFIG.overwriteExisting) {
    fail("File already exists:\n" + outputFile.fsName);
  }

  var options = new TiffSaveOptions();
  options.imageCompression = CONFIG.tiffCompression;
  options.layers = false;
  options.embedColorProfile = CONFIG.embedColorProfile;
  options.alphaChannels = true;

  doc.saveAs(outputFile, options, true, Extension.LOWERCASE);
}

function exportActiveDocument() {
  ensureOpenDocument();

  var sourceDoc = app.activeDocument;
  ensureSavedDocument(sourceDoc);

  var sourceFile = sourceDoc.fullName;
  var outputFolder = getOutputFolder(sourceFile);
  var outputFile = buildOutputFile(sourceFile, outputFolder);

  var workingDoc = sourceDoc.duplicate(sourceDoc.name + "_flat_export", false);
  app.activeDocument = workingDoc;

  try {
    workingDoc.flatten();
    resizeForPlacement(workingDoc);
    saveAsTiff(workingDoc, outputFile);
  } finally {
    workingDoc.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = sourceDoc;
  }

  alert("Exported:\n" + outputFile.fsName);
}

exportActiveDocument();
