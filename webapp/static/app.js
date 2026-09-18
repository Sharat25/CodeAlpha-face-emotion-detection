const CLASSES = window.__DEEPFER_CLASSES__ || [];

const EMOJI = {

  angry: "😠",

  disgust: "🤢",

  fear: "😨",

  happy: "😄",

  neutral: "😐",

  sad: "😢",

  surprise: "😲"

};

const COLOR_VAR = {

  angry: "--c-angry",

  disgust: "--c-disgust",

  fear: "--c-fear",

  happy: "--c-happy",

  neutral: "--c-neutral",

  sad: "--c-sad",

  surprise: "--c-surprise"

};

const root = document.documentElement;

const cssColor = (cls) =>

  getComputedStyle(root).getPropertyValue(COLOR_VAR[cls]).trim();


// ---------- elements ----------

const viewfinder = document.getElementById("viewfinder");

const previewImg = document.getElementById("preview-img");

const cameraVideo = document.getElementById("camera-video");

const overlayCanvas = document.getElementById("overlay-canvas");

const emptyState = document.getElementById("empty-state");

const fileInput = document.getElementById("file-input");

const cameraStartBtn = document.getElementById("camera-start");

const cameraStopBtn = document.getElementById("camera-stop");

const fpsReadout = document.getElementById("fps-readout");

const primaryEmoji = document.getElementById("primary-emoji");

const primaryLabel = document.getElementById("primary-label");

const primaryConfidence = document.getElementById("primary-confidence");

const faceCountEl = document.getElementById("face-count");

const spectrumEl = document.getElementById("spectrum");

const modeButtons = document.querySelectorAll(".mode-btn");

let mode = "upload";

let cameraStream = null;

let cameraLoopHandle = null;


// ---------- build spectrum rows once ----------

function buildSpectrum() {

  spectrumEl.innerHTML = "";

  CLASSES.forEach((cls) => {

    const row = document.createElement("div");

    row.className = "spectrum__row";

    row.innerHTML = `

      <div class = "spectrum__label">${cls}</div>

      <div class = "spectrum__track"><div class = "spectrum__fill" id = "fill-${cls}"></div></div>

      <div class = "spectrum__value" id = "value-${cls}">0%</div>

    `;

    spectrumEl.appendChild(row);

    const fill = row.querySelector(`#fill-${cls}`);

    fill.style.background = cssColor(cls);

  });

}

buildSpectrum();


// ---------- mode switching ----------

modeButtons.forEach((btn) => {btn.addEventListener("click", () => setMode(btn.dataset.mode));});

function setMode(newMode) {

  mode = newMode;

  modeButtons.forEach((b) => {

    const active = b.dataset.mode === mode;

    b.classList.toggle("is-active", active);

    b.setAttribute("aria-selected", active);

  });

  stopCamera();

  clearOverlay();

  resetReading();

  if (mode === "upload") {

    cameraVideo.hidden = true;

    cameraStartBtn.hidden = true;

    cameraStopBtn.hidden = true;

    fpsReadout.hidden = true;

    previewImg.hidden = !previewImg.src;

    emptyState.hidden = !!previewImg.src;

    viewfinder.onclick = () => fileInput.click();

  } else {

    previewImg.hidden = true;

    cameraStartBtn.hidden = false;

    emptyState.hidden = true;

    viewfinder.onclick = null;

  }

}


// ---------- upload flow ----------

viewfinder.onclick = () => fileInput.click();


fileInput.addEventListener("change", () => {

  const file = fileInput.files[0];

  if (file) loadImageFile(file);

});

viewfinder.addEventListener("dragover", (e) => {

  e.preventDefault();

});

viewfinder.addEventListener("drop", (e) => {

  e.preventDefault();

  if (mode !== "upload") return;

  const file = e.dataTransfer.files[0];

  if (file) loadImageFile(file);

});


function loadImageFile(file) {

  const reader = new FileReader();

  reader.onload = (e) => {

    previewImg.src = e.target.result;

    previewImg.hidden = false;

    emptyState.hidden = true;

    previewImg.onload = () => {

      sizeOverlayTo(previewImg);

      const form = new FormData();

      form.append("image", file);

      fetch("/predict", { method: "POST", body: form })

        .then((r) => r.json())

        .then((data) => handlePrediction(data, previewImg))

        .catch(() => resetReading("Error reaching server"));

    };

  };

  reader.readAsDataURL(file);

}


// ---------- camera flow ----------

cameraStartBtn.addEventListener("click", startCamera);

cameraStopBtn.addEventListener("click", stopCamera);

function startCamera() {

  navigator.mediaDevices

    .getUserMedia({ video: { width: 640, height: 480 } })

    .then((stream) => {

      cameraStream = stream;

      cameraVideo.srcObject = stream;

      cameraVideo.hidden = false;

      cameraStartBtn.hidden = true;

      cameraStopBtn.hidden = false;

      fpsReadout.hidden = false;

      viewfinder.classList.add("is-active");

      cameraVideo.onloadedmetadata = () => {

        sizeOverlayTo(cameraVideo);

        cameraLoop();

      };

    })

    .catch(() => {

      fpsReadout.hidden = false;

      fpsReadout.textContent = "Camera unavailable / permission denied";

    });

}

function stopCamera() {

  if (cameraLoopHandle) clearTimeout(cameraLoopHandle);

  cameraLoopHandle = null;

  if (cameraStream) {

    cameraStream.getTracks().forEach((t) => t.stop());

    cameraStream = null;

  }

  cameraVideo.hidden = true;

  cameraStartBtn.hidden = mode !== "camera";

  cameraStopBtn.hidden = true;

  viewfinder.classList.remove("is-active");

}

function cameraLoop() {

  const captureCanvas = document.createElement("canvas");

  captureCanvas.width = cameraVideo.videoWidth;

  captureCanvas.height = cameraVideo.videoHeight;

  const ctx = captureCanvas.getContext("2d");

  const tick = () => {

    if (!cameraStream) return;

    const t0 = performance.now();

    ctx.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

    fetch("/predict_frame", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({ image: dataUrl })

    })

      .then((r) => r.json())

      .then((data) => {

        handlePrediction(data, cameraVideo);

        const roundtrip = (performance.now() - t0).toFixed(0);

        fpsReadout.textContent = `${(1000 / (performance.now() - t0)).toFixed(1)} FPS · ${roundtrip} ms`;

      })

      .catch(() => {})

      .finally(() => {

        cameraLoopHandle = setTimeout(tick, 350); // throttled -- see README "Real-time performance notes"

      });

  };

  tick();

}


// ---------- shared rendering ----------

function sizeOverlayTo(mediaEl) {

  const rect = mediaEl.getBoundingClientRect();

  overlayCanvas.width = rect.width;

  overlayCanvas.height = rect.height;

  overlayCanvas.style.width = rect.width + "px";

  overlayCanvas.style.height = rect.height + "px";

}

function clearOverlay() {

  const ctx = overlayCanvas.getContext("2d");

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

}

function handlePrediction(data, mediaEl) {

  clearOverlay();

  if (!data || !data.faces || data.faces.length === 0) {

    faceCountEl.textContent = "No face detected";

    resetReading();

    return;

  }

  faceCountEl.textContent = `${data.faces.length} face${data.faces.length > 1 ? "s" : ""} detected · ${data.inference_ms} ms inference`;


  // natural media size vs displayed size, to scale box coordinates onto the overlay canvas
  const naturalW = mediaEl.videoWidth || mediaEl.naturalWidth;

  const naturalH = mediaEl.videoHeight || mediaEl.naturalHeight;

  const scaleX = overlayCanvas.width / naturalW;

  const scaleY = overlayCanvas.height / naturalH;

  const ctx = overlayCanvas.getContext("2d");

  data.faces.forEach((face) => {

    const color = cssColor(face.label);

    ctx.strokeStyle = color;

    ctx.lineWidth = 2;

    ctx.strokeRect(

      face.box.x * scaleX,

      face.box.y * scaleY,

      face.box.w * scaleX,

      face.box.h * scaleY

    );

    ctx.font = "500 13px 'IBM Plex Mono', monospace";

    ctx.fillStyle = color;

    ctx.fillText(

      `${face.label} ${(face.confidence * 100).toFixed(0)}%`,

      face.box.x * scaleX,

      face.box.y * scaleY - 6

    );

  });


  // primary reading = highest-confidence face

  const top = data.faces.reduce((a, b) =>
    
    a.confidence > b.confidence ? a : b,
  
  );
  
  primaryEmoji.textContent = EMOJI[top.label] || "—";
  
  primaryLabel.textContent = top.label;
  
  primaryLabel.style.color = cssColor(top.label);
  
  primaryConfidence.textContent = `${(top.confidence * 100).toFixed(1)}% confidence`;
  

  CLASSES.forEach((cls) => {
    
    const p = (top.probabilities[cls] || 0) * 100;
    
    const fill = document.getElementById(`fill-${cls}`);
    
    const value = document.getElementById(`value-${cls}`);
    
    if (fill) fill.style.width = `${p}%`;
    
    if (value) value.textContent = `${p.toFixed(0)}%`;
    
  });
  
}

function resetReading(message) {
  
  primaryEmoji.textContent = "—";
  
  primaryLabel.textContent = message || "Awaiting input";
  
  primaryLabel.style.color = "";
  
  primaryConfidence.innerHTML = "&nbsp;";
  
  faceCountEl.textContent = "";
  
  CLASSES.forEach((cls) => {
    
    const fill = document.getElementById(`fill-${cls}`);
    
    const value = document.getElementById(`value-${cls}`);
    
    if (fill) fill.style.width = "0%";
    
    if (value) value.textContent = "0%";
    
  });
  
}

resetReading();
