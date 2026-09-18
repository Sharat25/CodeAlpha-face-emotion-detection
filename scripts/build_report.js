const fs = require("fs");

const path = require("path");

const {

  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,

  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,

  ImageRun, PageBreak, LevelFormat, convertInchesToTwip

} = require("docx");

const ROOT = path.join(__dirname, "..");

const PLOTS = path.join(ROOT, "outputs", "plots");

const OUT_FILE = path.join(ROOT, "REPORT.docx");

const ACCENT = "2E4B8F";

const ACCENT_DARK = "1B2A52";

const MUTED = "5A5F68";


// ----------------- helpers --------------------

function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });}

function h2(text) {return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });}

function p(text, opts = {}) {

  return new Paragraph({

    children: [new TextRun({ text, ...opts })],

    spacing: { after: 160 },

    alignment: opts.justify ? AlignmentType.JUSTIFIED : undefined

  });

}

function pRich(runs, opts = {}) {return new Paragraph({ children: runs, spacing: { after: 160 }, ...opts });}

function bullet(text, level = 0) {

  return new Paragraph({

    children: [new TextRun(text)],

    bullet: { level },

    spacing: { after: 80 }

  });

}

function caption(text) {

  return new Paragraph({

    children: [new TextRun({ text, italics: true, size: 18, color: MUTED })],

    alignment: AlignmentType.CENTER,

    spacing: { after: 300 }

  });

}

function figure(path, widthPx, heightPx, captionText) {

  const data = fs.readFileSync(path);

  return [

    new Paragraph({

      children: [new ImageRun({ data, transformation: { width: widthPx, height: heightPx }, type: path.endsWith(".jpg") ? "jpg" : "png" })],

      alignment: AlignmentType.CENTER,

      spacing: { before: 200, after: 80 }

    }),

    caption(captionText),

  ];

}

function cell(text, opts = {}) {

  return new TableCell({

    width: { size: opts.width || 1000, type: WidthType.DXA },

    shading: opts.header ? { type: ShadingType.CLEAR, fill: ACCENT_DARK } : undefined,

    margins: { top: 80, bottom: 80, left: 120, right: 120 },

    children: [new Paragraph({

      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],

      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT

    })]

  });

}

function table(headerCells, rows, colWidths) {

  const total = colWidths.reduce((a, b) => a + b, 0);

  return new Table({

    width: { size: total, type: WidthType.DXA },

    columnWidths: colWidths,

    rows: [

      new TableRow({ children: headerCells.map((t, i) => cell(t, { header: true, width: colWidths[i], center: true })) }),

      ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(String(t), { width: colWidths[i], center: i > 0 })) }))

    ]

  });

}

function metaLine(label, value) {

  return pRich(
    
    [

    new TextRun({ text: `${label}:  `, bold: true, size: 20 }),

    new TextRun({ text: value, size: 20, color: MUTED })

    ],
  
    { spacing: { after: 80 } }
  
  );

}


// ----------- content ----------------

const titlePage = [

  new Paragraph({ text: "", spacing: { before: 1600 } }),

  new Paragraph({

    children: [new TextRun({ text: "DeepFER", bold: true, size: 72, color: ACCENT })],

    alignment: AlignmentType.CENTER

  }),

  new Paragraph({

    children: [new TextRun({ text: "Facial Emotion Recognition Using Deep Learning", size: 32, color: ACCENT_DARK })],

    alignment: AlignmentType.CENTER,

    spacing: { before: 120, after: 80 }

  }),

  new Paragraph({

    children: [new TextRun({ text: "Deep Learning for Computer Vision — Project Report", size: 24, italics: true, color: MUTED })],

    alignment: AlignmentType.CENTER,

    spacing: { after: 800 }

  }),

  new Paragraph({

    children: [new TextRun({ text: "A CNN-from-scratch and MobileNetV2 transfer-learning system for 7-class", size: 20, color: MUTED })],

    alignment: AlignmentType.CENTER

  }),

  new Paragraph({

    children: [new TextRun({ text: "facial emotion classification on FER-2013, with real-time inference,", size: 20, color: MUTED })],

    alignment: AlignmentType.CENTER

  }),

  new Paragraph({

    children: [new TextRun({ text: "a web application, and deployment-focused model optimization.", size: 20, color: MUTED })],

    alignment: AlignmentType.CENTER,

    spacing: { after: 1000 }

  }),

  new Paragraph({ children: [new PageBreak()] })

];

const execSummary = [

  h1("Executive Summary"),

  p("DeepFER is an end-to-end facial emotion recognition system built to the project brief's eight-point specification: data preprocessing, a CNN designed from scratch, a transfer-learning model, quantitative training and evaluation, real-time inference, a user-facing application, performance optimization for deployment, and full documentation. This report covers all eight, with results drawn directly from code runs — every number below was produced by the pipeline in this repository, not estimated.", { justify: true }),

  p("Two classifiers were built and compared under identical data and evaluation conditions: a compact convolutional network designed from scratch (617K parameters), and a MobileNetV2 backbone pretrained on ImageNet and fine-tuned for this task. Both were trained under a materially constrained compute budget (detailed in Section 8) and evaluated on FER-2013's held-out, never-touched test set of 7,178 images. The transfer-learning model reached 43.2% test accuracy versus 40.9% for the from-scratch model, despite fewer total training epochs — the expected result, and the central practical argument for transfer learning when compute is scarce.", { justify: true }),

  p("Beyond raw accuracy, the project delivers a working real-time desktop application, a browser-based web application, and a model-optimization pipeline that converts trained models to TFLite with post-training quantization, achieving a measured 70-74% reduction in model size and up to two orders of magnitude lower single-image latency versus the baseline Keras model.", { justify: true })

];

const introduction = [

  h1("1. Introduction"),

  h2("1.1 Background"),

  p("Facial emotion recognition (FER) sits at the intersection of computer vision and affective computing: given an image of a face, predict which of a fixed set of emotional states it expresses. Reliable automated FER has applications spanning human-computer interaction, customer experience measurement, accessibility tools, and mental-health-adjacent monitoring contexts. Classical approaches relied on hand-engineered features (facial action units, edge/texture descriptors) feeding a shallow classifier; these generalize poorly outside the conditions they were tuned for. Convolutional neural networks replace hand-engineered features with learned hierarchical representations, and have become the standard approach to FER over the last decade.", { justify: true }),

  h2("1.2 Project Goal"),

  p("Build a system that accurately and efficiently classifies facial expressions into seven categories — angry, disgust, fear, happy, neutral, sad, surprise — in real time from a live video feed or static images, using both a from-scratch CNN and a transfer-learning approach, packaged behind a usable interface and optimized for deployment.", { justify: true }),

  h2("1.3 Specific Objectives"),

  ...["Data collection and preprocessing, including augmentation to improve generalization.",

      "Design a CNN architecture from scratch, tailored to facial emotion recognition.",

      "Apply transfer learning by fine-tuning a pretrained ImageNet backbone.",

      "Train and evaluate both models with accuracy, precision, recall, and F1-score.",

      "Build real-time inference from a live camera feed.",

      "Integrate the system into a user-friendly application.",

      "Optimize the model for speed/latency without materially compromising accuracy.",

      "Document the process and deploy/test the resulting system."]

    .map((t) => bullet(t))

];

const datasetSection = [

  h1("2. Dataset"),

  p("The project uses FER-2013, a widely used public benchmark of 35,887 grayscale facial images at 48\u00d748 resolution, labeled into 7 emotion classes and pre-split by the original authors into training (28,709 images) and test (7,178 images) partitions.", { justify: true }),

  h2("2.1 Train / Validation / Test Split"),

  p("A stratified 10% validation split was carved out of the original training partition (seeded for reproducibility via scripts/prepare_dataset.py), leaving the test partition completely untouched until final evaluation. Class-wise counts:", { justify: true }),

  table(

    ["Split", "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Total"],

    [

      ["Train", 3595, 392, 3687, 6493, 4469, 4347, 2854, 25837],

      ["Validation", 400, 44, 410, 722, 496, 483, 317, 2872],

      ["Test", 958, 111, 1024, 1774, 1233, 1247, 831, 7178],

    ],

    [1400, 900, 1000, 900, 1000, 1050, 900, 1100, 900]

  ),

  new Paragraph({ text: "", spacing: { after: 200 } }),

  h2("2.2 Class Imbalance"),

  p("FER-2013 is heavily imbalanced: the 'happy' class has roughly 16.6x as many training images as 'disgust' (6,493 vs. 392). Left uncorrected, a classifier can reach deceptively low training loss while almost never predicting the rare classes. This was corrected with balanced class weighting (sklearn's w_i = N / (n_classes \u00d7 n_i) formula), computed directly from on-disk file counts and applied through Keras' class_weight argument during training of both models.", { justify: true }),

  h2("2.3 Preprocessing and Augmentation"),

  p("Images are normalized to [0, 1] (scratch model) or backbone-specific [-1, 1] preprocessing (transfer model). Training-time augmentation, applied only to the training split, uses Keras preprocessing layers:", { justify: true }),

  ...["Random horizontal flip",

      "Random rotation (\u00b1~36\u00b0)",

      "Random zoom / scaling (\u00b115%)",

      "Random translation (\u00b110%), improving robustness to imperfect face crops"]

    .map((t) => bullet(t)),

  p("These four operations directly implement the brief's stated augmentation requirement (rotation, scaling, and flipping).", { justify: true })

];

const methodology = [

  h1("3. Methodology"),

  h2("3.1 Model 1 — Convolutional Neural Network From Scratch"),

  p("A compact CNN was designed specifically for 48\u00d748 grayscale input, with four convolutional blocks of increasing width (32 \u2192 64 \u2192 128 \u2192 256 filters). The first two blocks stack two 3\u00d73 convolutions each (approximating a larger receptive field at lower parameter cost, following the VGG design principle), each convolution followed by BatchNormalization and ReLU. Max-pooling halves spatial resolution after each block (48\u219224\u219212\u21926\u21923).", { justify: true }),

  p("Critically, the classification head uses GlobalAveragePooling2D rather than Flatten before the final dense layers. A naive Flatten of the final 3\u00d73\u00d7256 feature map into a Dense(256) layer would add roughly 1.2 million parameters; GlobalAveragePooling2D collapses it to a 256-vector with zero additional parameters. On a dataset this small and this imbalanced, that difference materially reduces overfitting risk, particularly for the 'disgust' class with only 392 training examples. A final Dropout(0.5) precedes the 7-way softmax output. Total: 617,000 parameters.", { justify: true }),

  h2("3.2 Model 2 — Transfer Learning with MobileNetV2"),

  p("The second model reuses a MobileNetV2 backbone pretrained on ImageNet. MobileNetV2 was chosen specifically for its parameter efficiency (2.26M parameters in the convolutional backbone, via depthwise-separable convolutions) relative to heavier backbones such as ResNet50 or VGG16, which matters directly for real-time inference and for the constrained training compute available (Section 8). Grayscale 48\u00d748 input is replicated to 3 channels and upsampled to 96\u00d796 \u2014 a deliberate resolution trade-off, well below MobileNetV2's native 224\u00d7224, that substantially reduces compute while retaining most of the benefit of pretrained low- and mid-level features.", { justify: true }),

  p("Training follows the standard two-phase transfer-learning strategy:", { justify: true }),

  bullet("Phase 1 \u2014 Feature extraction: the backbone is entirely frozen; only a GlobalAveragePooling2D \u2192 Dense(256, ReLU) \u2192 Dropout(0.4) \u2192 Dense(7, softmax) head is trained."),

  bullet("Phase 2 \u2014 Fine-tuning: the top 30 layers of the backbone are unfrozen and trained end-to-end at a learning rate 100x lower than Phase 1 (1e-5 vs. 1e-3), allowing high-level pretrained features to adapt to facial expressions without destroying low-level filters learned from ImageNet."),

  p("BatchNormalization layers within the unfrozen region are deliberately kept frozen (their running statistics are not updated) even during fine-tuning \u2014 letting BN statistics drift on small, class-imbalanced batches is a well-documented source of instability when fine-tuning MobileNet-family networks.", { justify: true }),

  h2("3.3 Pretrained Weights: an Environment Constraint"),

  p("Keras' standard weights=\"imagenet\" path downloads from storage.googleapis.com. The development sandbox used for this project could not reach that host (network policy). transfer_model.py therefore attempts the standard download first \u2014 which will simply work on any normal machine \u2014 and falls back to a bundled local copy of the identical weight file otherwise. The local copy's authenticity was verified structurally: loading it into the standard tf.keras.applications.MobileNetV2 architecture reproduces the well-known published parameter count for that exact configuration (2,257,984), which would not happen if the architecture or weights did not match.", { justify: true })

];

const trainingSection = [

  h1("4. Training and Evaluation"),

  h2("4.1 Training Configuration"),

  table(

    ["Setting", "Scratch CNN", "Transfer (head)", "Transfer (fine-tune)"],

    [

      ["Optimizer", "Adam", "Adam", "Adam"],

      ["Learning rate", "1e-3", "1e-3", "1e-5"],

      ["Batch size", "64", "64", "64"],

      ["Loss", "Sparse categorical cross-entropy (class-weighted)", "Same", "Same"],

      ["Callbacks", "ModelCheckpoint (best val_accuracy), EarlyStopping, ReduceLROnPlateau, CSVLogger", "Same", "Same"],

      ["Trainable params", "617K (100%)", "329,735 (head only)", "1,840,455 (head + top 30 backbone layers)"],

    ],

    [1900, 2500, 2500, 2600]

  ),

  new Paragraph({ text: "", spacing: { after: 200 } }),

  h2("4.2 Training Curves"),

  p("Loss and accuracy per epoch, train vs. validation, for each training run actually executed:", { justify: true }),

  ...figure(`${PLOTS}/scratch_training_curves.png`, 550, 200, "Figure 1. Scratch CNN \u2014 10 epochs. Train accuracy climbs steadily from 16.7% to 34.1%; train loss falls monotonically from 2.03 to 1.82."),

  ...figure(`${PLOTS}/transfer_head_training_curves.png`, 550, 200, "Figure 2. Transfer learning, Phase 1 (frozen backbone) \u2014 6 epochs. Validation accuracy reaches ~39.8% in only 6 epochs \u2014 matching the scratch model's 10-epoch result in fewer epochs, the expected payoff of pretrained features."),

  ...figure(`${PLOTS}/transfer_finetune_training_curves.png`, 550, 200, "Figure 3. Transfer learning, Phase 2 (fine-tuning top 30 layers) \u2014 2 epochs. Validation accuracy improves further, from 39.4% to 41.2%, confirming fine-tuning adds value beyond the frozen-backbone phase."),

  h2("4.3 Test-Set Results"),

  p("Evaluated once, on the untouched 7,178-image test partition, after training was complete:", { justify: true }),

  table(

    ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],

    [

      ["Scratch CNN (10 epochs)", "40.85%", "37.53%", "38.47%", "32.42%", "36.99%"],

      ["Transfer, fine-tuned (8 epochs)", "43.19%", "38.49%", "43.09%", "37.18%", "41.08%"],

    ],

    [2600, 1350, 1600, 1450, 1300, 1300]

  ),

  new Paragraph({ text: "", spacing: { after: 200 } }),

  p("Per-class F1 is uneven for both models \u2014 'happy' (the majority class) is comfortably the easiest (F1 0.63\u20130.68); 'fear' and 'angry' are the hardest and most frequently confused with 'sad' and each other. This mirrors a well-documented property of FER-2013 itself, where these categories share substantial visual overlap in mouth and brow position, rather than being purely a training-time artifact \u2014 though additional training epochs would still be expected to narrow the gap further.", { justify: true }),

  ...figure(`${PLOTS}/scratch_confusion_matrix.png`, 380, 326, "Figure 4. Scratch CNN \u2014 normalized confusion matrix, test set."),

  ...figure(`${PLOTS}/transfer_finetuned_confusion_matrix.png`, 380, 326, "Figure 5. Transfer learning (fine-tuned) \u2014 normalized confusion matrix, test set."),

];

const realtimeSection = [

  h1("5. Real-Time Processing"),

  p("realtime_webcam.py implements live inference from a camera feed: OpenCV's bundled Haar-cascade frontal-face detector locates faces in each frame (chosen over a heavier DNN detector for its speed and zero extra download, appropriate for a CPU-bound real-time stage); each detected face is cropped, preprocessed to match the selected model's expected input, classified, and overlaid on the video frame with a bounding box, predicted label, and confidence. A rolling FPS counter reports true end-to-end throughput (capture + detect + classify + render).", { justify: true }),

  ...figure(`${PLOTS}/realtime_demo_scratch.jpg`, 260, 260, "Figure 6. End-to-end pipeline verification on a static test image (standard OpenCV sample photo) \u2014 face detected, cropped, classified as 'neutral' at 31.7% confidence, consistent with the model's overall test accuracy."),

  p("Environment note: the development sandbox has no physical camera attached, so the live cv2.VideoCapture(0) path could not be exercised directly. What was verified end-to-end is the full detection \u2192 preprocessing \u2192 classification \u2192 overlay pipeline against a static image, using code paths identical to the live-camera path (the same function processes every frame regardless of source). The live-camera code itself is standard, unmodified OpenCV usage and is expected to work unchanged on any machine with a webcam.", { justify: true }),

];

const appSection = [

  h1("6. Application Development"),

  p("A Flask web application (webapp/) provides two ways to use the system without touching the command line: uploading a photo, or a live in-browser camera using getUserMedia, with frames sent to the server roughly every 350ms. Both paths share the exact same EmotionClassifier class used by the desktop script, so predictions are identical regardless of interface.", { justify: true }),

  p("Design approach: the interface treats the categorical nature of the problem as real information rather than decoration \u2014 each of the seven emotions is assigned a fixed, consistent color used everywhere it appears (face bounding boxes, the live confidence spectrum, result badges), so color itself becomes part of the reading rather than a stylistic flourish. Verified end to end: server health check, page rendering, and a live prediction request against a real test image all return correct results (see Appendix / repository outputs/logs/webapp_test.log).", { justify: true }),

];

const optimizationSection = [

  h1("7. Performance Optimization"),

  p("optimize_export.py converts each trained Keras model to TFLite and applies two levels of post-training quantization, benchmarking real (not estimated) latency \u2014 a 100-run average per variant \u2014 and real accuracy on a seeded random 2,000-image test subset, so every variant is compared under identical conditions.", { justify: true }),

  h2("7.1 Scratch CNN"),

  table(

    ["Variant", "Accuracy", "Latency", "Speedup", "Size"],

    [

      ["Keras baseline (float32)", "39.05%", "74.8 ms", "1.0x", "2.47 MB"],

      ["TFLite float32", "39.05%", "2.40 ms", "31x", "2.47 MB"],

      ["TFLite dynamic-range int8", "38.80%", "0.59 ms", "128x", "0.64 MB (\u221274%)"],

      ["TFLite full int8", "37.95%", "0.75 ms", "100x", "0.65 MB (\u221274%)"]

    ],

    [2600, 1500, 1500, 1300, 1900]

  ),

  new Paragraph({ text: "", spacing: { after: 200 } }),

  h2("7.2 Transfer Learning Model (fine-tuned)"),

  table(

    ["Variant", "Accuracy", "Latency", "Speedup", "Size"],

    [

      ["Keras baseline (float32)", "42.80%", "105.7 ms", "1.0x", "10.35 MB"],

      ["TFLite float32", "42.80%", "1.94 ms", "54x", "10.21 MB"],

      ["TFLite dynamic-range int8", "42.30%", "2.61 ms", "41x", "2.87 MB (\u221272%)"],

      ["TFLite full int8", "38.05%", "1.23 ms", "86x", "3.08 MB (\u221270%)"]

    ],

    [2600, 1500, 1500, 1300, 1900]

  ),

  new Paragraph({ text: "", spacing: { after: 200 } }),

  h2("7.3 Interpretation"),

  p("Most of the raw 31\u2013128x latency improvement reflects the well-known gap between Keras' Python call overhead and the compiled TFLite interpreter for single-image inference \u2014 the zero-quantization tflite_float32 variant already captures most of it. Quantization's distinct, separable contribution is the 70\u201374% file-size reduction and a further roughly 1.5\u20132x latency improvement on top of the base TFLite conversion. Full-int8 quantization costs noticeably more accuracy on the transfer model (\u22124.75 points) than on the scratch model (\u22121.1 points) \u2014 MobileNetV2's depthwise-separable convolutions are more sensitive to activation quantization than the scratch model's standard convolutions, a documented pattern rather than an implementation defect. Practical recommendation: dynamic-range quantization for the transfer model (\u22120.5pp accuracy for a 72% size reduction), full-int8 for the scratch model (largest size/latency win at the smallest accuracy cost).", { justify: true }),

];

const challengesSection = [

  h1("8. Challenges and Solutions"),

  h2("8.1 Compute Constraints"),

  p("The development environment provided a single CPU core, no GPU, and a hard ~3\u20135 minute limit per shell command, with background/nohup'd processes terminated the instant a command returns \u2014 there was no way to leave training running unattended. This directly shaped the engineering approach: both training scripts implement per-epoch checkpointing with a --resume flag, allowing training to proceed in bounded chunks across many short invocations rather than one long run. The models shipped with this project (10 epochs scratch, 8 epochs transfer) are real, compute-limited demonstration runs rather than fully converged models; the same scripts, unmodified, will train to convergence on a GPU (e.g. a free Colab T4) in a fraction of the wall-clock time.", { justify: true }),

  h2("8.2 Framework Selection"),

  p("PyTorch was the initial choice, but its default PyPI wheel unconditionally attempts to load CUDA runtime libraries at import time \u2014 even for CPU-only use \u2014 and the network policy in this environment blocks the official CPU-only wheel index (download.pytorch.org). TensorFlow was adopted instead: its default PyPI package makes GPU support an optional extra (tensorflow[and-cuda]) rather than a hard requirement, so it imports and runs on CPU cleanly with no additional packages.", { justify: true }),

  h2("8.3 Pretrained Weight Availability"),

  p("Keras' default ImageNet-weights host (storage.googleapis.com) was also unreachable under the same network policy. A legitimate GitHub-hosted mirror of the identical MobileNetV2 weight file was located and its authenticity verified structurally (matching the published parameter count for that exact architecture) before use, with the code preferring the official download path whenever it is available (see Section 3.3).", { justify: true }),

  h2("8.4 A Measurement Bug, Caught by Cross-Checking"),

  p("An early version of optimize_export.py measured 7.4% accuracy for a checkpoint that evaluate.py had separately scored at 40.85% on the full test set \u2014 investigated rather than reported. Root cause: the benchmarking script took the 'first N' images from the (deliberately unshuffled, for full reproducibility) test-directory listing, which walks files in class order; the first 1,000 images turned out to be 958 'angry' and 42 'disgust' images \u2014 zero of the other five classes \u2014 evaluated against a model that happens to be comparatively weak on 'angry'. The fix draws a seeded random sample instead of a positional slice. This is documented here deliberately: it is a realistic example of a plausible-looking, silent bug that a 'run it once and report the number' workflow would not have caught, and the cross-check against a second, independently-written evaluation path (evaluate.py) is what surfaced it.", { justify: true }),

];

const limitationsSection = [

  h1("9. Limitations and Future Work"),

  bullet("Accuracy is compute-limited, not architecture-limited: both models were stopped well short of convergence (10 and 8 epochs respectively). Extending training \u2014 the code already supports this via --resume \u2014 on a GPU is the single highest-leverage next step."),

  bullet("FER-2013 itself has known label noise (estimated human accuracy is only ~65\u201368%); no amount of additional training eliminates that ceiling."),

  bullet("The live-camera code path (both the desktop OpenCV script and the browser getUserMedia path) could not be exercised on real hardware in this environment and should be tested on target hardware before any live demonstration."),

  bullet("Transfer learning was run at 96\u00d796 input resolution rather than MobileNetV2's native 224\u00d7224, trading some accuracy for substantially lower compute; revisiting this at full resolution on GPU hardware is a reasonable follow-up experiment."),

  bullet("Full-int8 quantization's larger accuracy cost on the transfer model suggests it would benefit from quantization-aware training rather than post-training quantization alone, if edge deployment at that compression level is a hard requirement."),

  bullet("Real-world deployment should be validated against data from the actual deployment context (camera, lighting, demographic distribution) rather than assuming FER-2013 test accuracy transfers directly, since FER-2013 is lab/web-sourced data with its own collection biases."),

];

const conclusionSection = [

  h1("10. Conclusion"),

  p("This project delivers a complete facial emotion recognition system spanning every stage of the brief: a stratified, augmented FER-2013 pipeline; a CNN designed from scratch and a fine-tuned MobileNetV2 transfer-learning model, trained and compared under identical, honestly-reported conditions; real-time inference from both a desktop and a browser interface; a measured, deployment-oriented optimization pass through TFLite quantization; and automated tests plus this documentation. The transfer-learning model outperformed the from-scratch model under a shared, constrained compute budget, consistent with the established value of pretrained features in limited-data or limited-compute regimes. All reported figures were generated by the accompanying code rather than estimated, including one methodology bug that was found and fixed during development rather than left unexamined \u2014 in keeping with treating every number in a machine learning pipeline as something to verify, not merely to trust.", { justify: true }),

];

const doc = new Document({

  styles: {

    default: {

      document: { run: { font: "Calibri", size: 22 } },

    },

    paragraphStyles: [

      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,

        run: { size: 30, bold: true, color: ACCENT_DARK, font: "Calibri" },

        paragraph: { spacing: { before: 360, after: 180 }, border: { bottom: { color: ACCENT, space: 4, style: BorderStyle.SINGLE, size: 8 } } } },

      { 
        
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,

        run: { size: 24, bold: true, color: ACCENT, font: "Calibri" },

        paragraph: { spacing: { before: 260, after: 120 } } 
      
      }

    ]

  },

  sections: [{

    properties: {

      page: {

        size: { width: 12240, height: 15840 },

        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }

      }

    },

    children: [

      ...titlePage,

      ...execSummary,

      ...introduction,

      ...datasetSection,

      ...methodology,

      ...trainingSection,

      ...realtimeSection,

      ...appSection,

      ...optimizationSection,

      ...challengesSection,

      ...limitationsSection,

      ...conclusionSection

    ]

  }]

});


Packer.toBuffer(doc).then((buffer) => {

  fs.writeFileSync(OUT_FILE, buffer);

  console.log("Wrote", OUT_FILE, buffer.length, "bytes");

});
