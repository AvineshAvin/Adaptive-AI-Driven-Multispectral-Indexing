// State variables
let currentQueryId = null;
let currentQueryClass = null;
let allSamples = {};
let currentPolicy = 'recall_aware';

// Chart instances
let spectralChartInstance = null;
let classTierChartInstance = null;
let benchPrecisionChartInstance = null;
let benchEfficiencyChartInstance = null;
let ablationChartInstance = null;
let spectralSignaturesChartInstance = null;
let simProgressionChartInstance = null;
let simHotClassChartInstance = null;
let paretoChartInstance = null;
let scalabilityChartInstance = null;

// Bands labels
const BAND_LABELS = [
  "B01 (443nm)", "B02 (490nm)", "B03 (560nm)", "B04 (665nm)",
  "B05 (705nm)", "B06 (740nm)", "B07 (783nm)", "B08 (842nm)",
  "B8A (865nm)", "B09 (940nm)", "B10 (1375nm)", "B11 (1610nm)", "B12 (2190nm)"
];

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) lucide.createIcons();
  await loadStatus();
  await loadSamples();
  await loadBenchmarkData();
  await loadSpectralSignatures();
  initSpectralChart();
});

// Navigation Tab Switcher
function switchTab(tabId) {
  const tabs = ['retrieval', 'tiers', 'benchmarks', 'ablation', 'simulation'];
  tabs.forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (!el || !btn) return;
    if (t === tabId) {
      el.classList.remove('hidden');
      btn.className = 'nav-tab active-tab flex items-center gap-2 py-2 px-4 text-xs font-bold text-white bg-slate-900 rounded-full shadow-sm';
    } else {
      el.classList.add('hidden');
      btn.className = 'nav-tab text-slate-600 hover:text-slate-900 hover:bg-purple-50/80 flex items-center gap-2 py-2 px-4 text-xs font-semibold rounded-full';
    }
  });

  if (tabId === 'tiers') loadTiersOverview();
  if (tabId === 'benchmarks') loadBenchmarkData();
  if (tabId === 'ablation') loadSpectralSignatures();
  if (window.lucide) lucide.createIcons();
}

// Adaptive Policy Selector
function setPolicyMode(mode) {
  currentPolicy = mode;
  const btnRecall = document.getElementById('btn-policy-recall');
  const btnAgg = document.getElementById('btn-policy-agg');
  const label = document.getElementById('policy-label');
  if (mode === 'recall_aware') {
    btnRecall.className = 'py-1 px-2 rounded-lg font-bold bg-purple-600 text-white shadow-sm transition-all text-center';
    btnAgg.className = 'py-1 px-2 rounded-lg font-medium bg-purple-50 text-slate-700 hover:bg-purple-100 transition-all text-center';
    label.innerText = 'Recall-Aware (6/10+)';
  } else {
    btnAgg.className = 'py-1 px-2 rounded-lg font-bold bg-purple-600 text-white shadow-sm transition-all text-center';
    btnRecall.className = 'py-1 px-2 rounded-lg font-medium bg-purple-50 text-slate-700 hover:bg-purple-100 transition-all text-center';
    label.innerText = 'Aggressive (88% Pruned)';
  }
}

// Alpha slider display
function updateAlphaSlider(val) {
  const label = document.getElementById('alpha-val');
  label.innerText = `Manual: ${parseFloat(val).toFixed(2)}`;
  document.getElementById('query-alpha-tag').innerText = `Manual α: ${parseFloat(val).toFixed(2)}`;
}

// 1. Load System Status
async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('header-repo-size').innerText = `${data.repository_size.toLocaleString()} GeoTIFFs`;
    const hotCount = data.tier_distribution.HOT || 0;
    const warmCount = data.tier_distribution.WARM || 0;
    const coldCount = data.tier_distribution.COLD || 0;
    document.getElementById('header-hot-count').innerText = `${hotCount} HOT / ${warmCount} WARM`;
  } catch (e) {
    console.error("Status load error:", e);
  }
}

// 2. Load Samples and setup pills
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    allSamples = await res.json();
    const pillsContainer = document.getElementById('class-pills');
    pillsContainer.innerHTML = '';

    const classes = Object.keys(allSamples);
    classes.forEach((c, idx) => {
      const btn = document.createElement('button');
      btn.className = `px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
        idx === 0 
          ? 'bg-purple-600 text-white shadow-sm shadow-purple-500/25' 
          : 'bg-purple-50 text-slate-700 hover:bg-purple-100 hover:text-slate-900 border border-purple-100'
      }`;
      btn.innerText = c;
      btn.onclick = () => selectClassPill(c, btn);
      pillsContainer.appendChild(btn);
    });

    if (classes.length > 0) {
      selectClassPill(classes[0], pillsContainer.firstChild);
    }
  } catch (e) {
    console.error("Samples load error:", e);
  }
}

function selectClassPill(className, activeBtn) {
  const allBtns = document.getElementById('class-pills').children;
  for (let b of allBtns) {
    b.className = 'px-3 py-1.5 rounded-full text-xs font-medium bg-purple-50 text-slate-700 hover:bg-purple-100 hover:text-slate-900 border border-purple-100 transition-all';
  }
  activeBtn.className = 'px-3 py-1.5 rounded-full text-xs font-semibold bg-purple-600 text-white shadow-sm shadow-purple-500/25 transition-all';

  const container = document.getElementById('sample-thumbnails');
  container.innerHTML = '';
  const samples = allSamples[className] || [];

  samples.forEach((s, i) => {
    const imgEl = document.createElement('img');
    imgEl.src = s.preview_url;
    imgEl.alt = s.id;
    imgEl.title = `${s.class_name} (${s.id})`;
    imgEl.className = 'w-12 h-12 rounded-xl object-cover cursor-pointer border-2 border-transparent hover:border-purple-600 transition-all shadow-sm';
    imgEl.onclick = () => selectQueryImage(s.id, s.class_name, s.preview_url);
    container.appendChild(imgEl);
  });

  if (samples.length > 0) {
    selectQueryImage(samples[0].id, samples[0].class_name, samples[0].preview_url);
  }
}

function selectQueryImage(id, className, previewUrl) {
  currentQueryId = id;
  currentQueryClass = className;
  uploadedFileObject = null;
  document.getElementById('uploaded-badge').classList.add('hidden');
  document.getElementById('current-query-id').innerText = `${id}.tif`;
  document.getElementById('current-query-class').innerText = className;
  document.getElementById('current-query-img').src = previewUrl;
  document.getElementById('current-query-desc').innerText = `13 Sentinel-2 Spectral Bands loaded & preprocessed`;
  document.getElementById('query-band-tag').innerText = `True Color RGB`;
}

// Upload & Query Mode Switcher
let uploadedFileObject = null;
let currentCirUrl = null;
let currentTrueColorUrl = null;
let isCirView = false;

function switchQueryMode(mode) {
  const panelDataset = document.getElementById('panel-dataset-explorer');
  const panelUpload = document.getElementById('panel-upload-image');
  const btnDataset = document.getElementById('btn-mode-dataset');
  const btnUpload = document.getElementById('btn-mode-upload');
  const subtitle = document.getElementById('query-mode-subtitle');

  if (mode === 'upload') {
    panelDataset.classList.add('hidden');
    panelUpload.classList.remove('hidden');
    btnUpload.className = 'px-3.5 py-1.5 rounded-full text-xs font-bold bg-slate-900 text-white shadow-sm flex items-center gap-1.5 transition-all';
    btnDataset.className = 'px-3.5 py-1.5 rounded-full text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-purple-50 flex items-center gap-1.5 transition-all';
    subtitle.innerText = 'Upload GeoTIFF (.tif) or RGB (.jpg/.png) to analyze';
  } else {
    panelDataset.classList.remove('hidden');
    panelUpload.classList.add('hidden');
    btnDataset.className = 'px-3.5 py-1.5 rounded-full text-xs font-bold bg-slate-900 text-white shadow-sm flex items-center gap-1.5 transition-all';
    btnUpload.className = 'px-3.5 py-1.5 rounded-full text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-purple-50 flex items-center gap-1.5 transition-all';
    subtitle.innerText = 'Choose from 10 Land-Cover Classes';
  }
  if (window.lucide) lucide.createIcons();
}

// Drag & Drop Handlers
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  const dz = document.getElementById('dropzone');
  dz.classList.add('border-purple-500', 'bg-purple-50/50');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  const dz = document.getElementById('dropzone');
  dz.classList.remove('border-purple-500', 'bg-purple-50/50');
}

function handleFileDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  const dz = document.getElementById('dropzone');
  dz.classList.remove('border-purple-500', 'bg-purple-50/50');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    processSelectedFile(e.dataTransfer.files[0]);
  }
}

function handleFileSelect(e) {
  if (e.target.files && e.target.files.length > 0) {
    processSelectedFile(e.target.files[0]);
  }
}

function processSelectedFile(file) {
  uploadedFileObject = file;
  currentQueryId = null;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    if (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')) {
      document.getElementById('current-query-img').src = 'https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=128&q=80';
    } else {
      document.getElementById('current-query-img').src = e.target.result;
    }
  };
  reader.readAsDataURL(file);

  document.getElementById('uploaded-badge').classList.remove('hidden');
  document.getElementById('current-query-id').innerText = file.name;
  document.getElementById('current-query-class').innerText = "Analyzing Scene...";
  document.getElementById('current-query-desc').innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB) - Ready for Analysis`;

  uploadAndAnalyzeFile(file);
}

function loadSampleToAnalyze(sampleId) {
  const parts = sampleId.split('_');
  const cls = parts[0];
  selectQueryImage(sampleId, cls, `/api/image/${sampleId}`);
  executeRetrieval();
}

// Upload & Analyze API call
async function uploadAndAnalyzeFile(file) {
  const mode = document.querySelector('input[name="retrieval-mode"]:checked').value;
  const topK = parseInt(document.getElementById('top-k-slider').value);
  const alphaVal = document.getElementById('alpha-slider').value;
  const tauVal = parseFloat(document.getElementById('threshold-slider').value);
  const probeVal = parseInt(document.getElementById('probe-slider').value);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', mode);
  formData.append('top_k', topK);
  formData.append('early_stop_threshold', tauVal);
  formData.append('early_stop_policy', currentPolicy);
  formData.append('n_probe', probeVal);
  if (!document.getElementById('alpha-val').innerText.includes('Auto')) {
    formData.append('alpha', parseFloat(alphaVal));
  }

  const btn = document.getElementById('btn-execute-search');
  btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin text-white"></i> Analyzing Scene...';
  if (window.lucide) lucide.createIcons();

  const grid = document.getElementById('results-grid');
  grid.innerHTML = '<div class="col-span-full py-16 text-center text-slate-500 text-xs flex items-center justify-center gap-2"><i data-lucide="loader-2" class="w-4 h-4 animate-spin text-purple-600"></i> Analyzing 13 spectral bands & retrieving matching remote sensing candidates...</div>';
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/upload_and_analyze', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || "Analysis failed");
    }
    const data = await res.json();

    renderAnalysisCard(data.analysis);

    document.getElementById('current-query-class').innerText = data.analysis.predicted_class;
    document.getElementById('current-query-desc').innerText = data.analysis.channels_info;
    if (data.analysis.preview_url) {
      document.getElementById('current-query-img').src = data.analysis.preview_url;
    }
    document.getElementById('query-alpha-tag').innerText = `Dynamic α: ${data.analysis.estimated_alpha}`;

    renderRetrievalResults({
      query_class: data.analysis.predicted_class,
      query_band_signature: data.analysis.band_signature,
      top_retrieved_band_signature: data.results.length > 0 ? (data.analysis.band_signature) : [],
      results: data.results,
      stats: data.stats
    });

  } catch (e) {
    console.error("Upload & Analysis error:", e);
    grid.innerHTML = `<div class="col-span-full py-10 text-center text-rose-500 text-xs">Error during analysis: ${e.message}</div>`;
  } finally {
    btn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i> <span>Analyze & Search</span>';
    if (window.lucide) lucide.createIcons();
  }
}

// Render Analysis Report Card
function renderAnalysisCard(analysis) {
  const card = document.getElementById('image-analysis-card');
  card.classList.remove('hidden');

  currentTrueColorUrl = analysis.preview_url;
  currentCirUrl = analysis.cir_preview_url;
  isCirView = false;

  document.getElementById('analysis-preview-img').src = analysis.preview_url;
  document.getElementById('analysis-predicted-class').innerText = analysis.predicted_class;
  document.getElementById('analysis-confidence-badge').innerText = `${analysis.classification_confidence}% Confidence`;
  document.getElementById('analysis-channels-text').innerText = analysis.channels_info;
  document.getElementById('analysis-dims-text').innerText = analysis.dimensions;
  document.getElementById('analysis-alpha-badge').innerText = `Recommended α = ${analysis.estimated_alpha}`;

  const matchBadge = document.getElementById('analysis-match-badge');
  if (analysis.is_dataset_image && analysis.matched_dataset_id) {
    matchBadge.innerHTML = `<span class="text-emerald-700 font-semibold font-mono">✅ EuroSAT Dataset Item: ${analysis.matched_dataset_id}.tif</span>`;
  } else {
    matchBadge.innerHTML = `<span class="text-purple-700 font-semibold">🌐 Uploaded Satellite Image (Classified as ${analysis.predicted_class})</span>`;
  }

  const ndvi = analysis.spectral_indices.ndvi;
  const ndwi = analysis.spectral_indices.ndwi;
  const ndbi = analysis.spectral_indices.ndbi;

  document.getElementById('analysis-ndvi-val').innerText = (ndvi.value >= 0 ? '+' : '') + ndvi.value;
  document.getElementById('analysis-ndvi-interp').innerText = ndvi.interpretation;

  document.getElementById('analysis-ndwi-val').innerText = (ndwi.value >= 0 ? '+' : '') + ndwi.value;
  document.getElementById('analysis-ndwi-interp').innerText = ndwi.interpretation;

  document.getElementById('analysis-ndbi-val').innerText = (ndbi.value >= 0 ? '+' : '') + ndbi.value;
  document.getElementById('analysis-ndbi-interp').innerText = ndbi.interpretation;

  document.getElementById('analysis-edge-val').innerText = analysis.spatial_metrics.edge_energy;
  document.getElementById('analysis-texture-interp').innerText = analysis.spatial_metrics.texture_complexity;

  const cirBtn = document.getElementById('btn-toggle-cir');
  if (analysis.cir_preview_url) {
    cirBtn.classList.remove('hidden');
    cirBtn.innerText = 'CIR NIR';
  } else {
    cirBtn.classList.add('hidden');
  }

  if (window.lucide) lucide.createIcons();
}

function toggleCompositeView() {
  const imgEl = document.getElementById('analysis-preview-img');
  const btn = document.getElementById('btn-toggle-cir');
  if (!currentCirUrl) return;

  if (isCirView) {
    imgEl.src = currentTrueColorUrl;
    btn.innerText = 'CIR NIR';
    btn.className = 'absolute -bottom-2 -right-2 px-2.5 py-0.5 rounded-full bg-purple-600 text-[10px] font-bold text-white shadow-sm hover:bg-purple-700 transition-colors';
    isCirView = false;
  } else {
    imgEl.src = currentCirUrl;
    btn.innerText = 'RGB TRUE';
    btn.className = 'absolute -bottom-2 -right-2 px-2.5 py-0.5 rounded-full bg-emerald-600 text-[10px] font-bold text-white shadow-sm hover:bg-emerald-700 transition-colors';
    isCirView = true;
  }
}

// 3. Execute Retrieval
async function executeRetrieval() {
  if (uploadedFileObject) {
    return uploadAndAnalyzeFile(uploadedFileObject);
  }

  if (!currentQueryId) {
    alert("Please select or upload a query image first.");
    return;
  }

  const mode = document.querySelector('input[name="retrieval-mode"]:checked').value;
  const topK = parseInt(document.getElementById('top-k-slider').value);
  const alphaVal = document.getElementById('alpha-slider').value;
  const tauVal = parseFloat(document.getElementById('threshold-slider').value);
  const probeVal = parseInt(document.getElementById('probe-slider').value);

  const payload = {
    image_id: currentQueryId,
    mode: mode,
    top_k: topK,
    alpha: document.getElementById('alpha-val').innerText.includes('Auto') ? null : parseFloat(alphaVal),
    early_stop_threshold: tauVal,
    early_stop_policy: currentPolicy,
    n_probe: probeVal
  };

  const grid = document.getElementById('results-grid');
  grid.innerHTML = '<div class="col-span-full py-16 text-center text-slate-500 text-xs flex items-center justify-center gap-2"><i data-lucide="loader-2" class="w-4 h-4 animate-spin text-purple-600"></i> Searching index across 13 spectral bands...</div>';
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderRetrievalResults(data);
  } catch (e) {
    console.error("Retrieval error:", e);
    grid.innerHTML = `<div class="col-span-full py-10 text-center text-rose-500 text-xs">Error during search: ${e.message}</div>`;
  }
}

function renderRetrievalResults(data) {
  const stats = data.stats;
  const results = data.results;

  // Update telemetry bar
  document.getElementById('stat-latency').innerText = `${parseFloat(stats.search_time_ms).toFixed(3)} ms`;
  document.getElementById('stat-reduction').innerText = `${parseFloat(stats.search_space_reduction_pct).toFixed(1)}%`;
  document.getElementById('stat-cands').innerText = `${stats.candidates_evaluated} / ${stats.total_repository_size || 1500} scanned`;
  document.getElementById('stat-confidence').innerText = `${(parseFloat(stats.aggregate_confidence || 0) * 100).toFixed(1)}%`;
  document.getElementById('stat-tierpath').innerText = stats.tier_status || stats.mode;
  document.getElementById('stat-earlystop').innerText = stats.tier_path || 'Direct index scan';

  // Precision matches in top-k
  const matches = results.filter(r => r.class_name === data.query_class).length;
  document.getElementById('stat-accuracy').innerText = `${matches} / ${results.length}`;

  // Speedup tag
  if (stats.mode.includes("Adaptive")) {
    document.getElementById('stat-speedup').innerText = `${stats.search_space_reduction_pct}% space pruned`;
  } else if (stats.mode.includes("Inverted")) {
    document.getElementById('stat-speedup').innerText = `${stats.search_space_reduction_pct}% space pruned`;
  } else {
    document.getElementById('stat-speedup').innerText = `Baseline (1.0x)`;
  }

  // Update Spectral Chart
  updateSpectralChart(data.query_band_signature, data.top_retrieved_band_signature, data.query_class);

  // Render Candidate Cards
  const grid = document.getElementById('results-grid');
  grid.innerHTML = '';

  if (results.length === 0) {
    grid.innerHTML = '<div class="col-span-full py-10 text-center text-slate-500 text-xs">No matching candidates found.</div>';
    return;
  }

  results.forEach((item, idx) => {
    const isMatch = item.class_name === data.query_class;
    const tierBadge = item.tier === 'HOT' ? '<span class="text-[9px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200 font-bold">🔥 HOT</span>' :
                      item.tier === 'WARM' ? '<span class="text-[9px] px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 border border-purple-200 font-bold">🟣 WARM</span>' :
                      '<span class="text-[9px] px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 border border-blue-200 font-bold">❄️ COLD</span>';

    const card = document.createElement('div');
    card.className = 'candidate-card bg-white border border-purple-100 hover:border-purple-300 rounded-2xl p-3 flex flex-col justify-between shadow-sm hover:shadow-md transition-all';
    card.innerHTML = `
      <div>
        <div class="relative mb-2">
          <img src="${item.preview_url}" alt="${item.id}" class="w-full h-24 object-cover rounded-xl border border-purple-50">
          <span class="absolute top-1.5 left-1.5 px-2 py-0.5 rounded-full bg-slate-900/80 backdrop-blur text-[10px] font-mono font-bold text-white shadow-sm">#${idx + 1}</span>
          <div class="absolute top-1.5 right-1.5">${tierBadge}</div>
        </div>

        <div class="flex items-center justify-between mb-1">
          <span class="text-xs font-bold text-slate-800 truncate">${item.class_name}</span>
          ${isMatch ? '<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-600"></i>' : '<i data-lucide="help-circle" class="w-3.5 h-3.5 text-amber-500"></i>'}
        </div>

        <div class="space-y-1 text-[10px] font-mono text-slate-500">
          <div class="flex justify-between">
            <span>Combined Sim:</span>
            <span class="font-bold text-purple-700">${(item.score * 100).toFixed(1)}%</span>
          </div>
          <div class="flex justify-between">
            <span>Confidence:</span>
            <span class="text-amber-600 font-medium">${item.confidence ? (item.confidence * 100).toFixed(1) + '%' : '--'}</span>
          </div>
          <div class="flex justify-between text-[9px] text-slate-400">
            <span>Spectral: ${(item.spectral_score ? (item.spectral_score * 100).toFixed(0) : '--')}%</span>
            <span>Spatial: ${(item.spatial_score ? (item.spatial_score * 100).toFixed(0) : '--')}%</span>
          </div>
        </div>
      </div>

      <div class="mt-2 pt-2 border-t border-purple-50 flex items-center justify-between">
        <span class="text-[9px] text-slate-400 font-mono">Act: ${item.activity_score || '0.3'}</span>
        <div class="flex items-center gap-1">
          <button onclick="submitFeedback('${item.id}', true, this)" title="Mark Relevant (+1)" class="p-1 rounded-lg hover:bg-emerald-50 text-slate-400 hover:text-emerald-600 transition-colors">
            <i data-lucide="thumbs-up" class="w-3 h-3"></i>
          </button>
          <button onclick="submitFeedback('${item.id}', false, this)" title="Mark Irrelevant (-0.5)" class="p-1 rounded-lg hover:bg-rose-50 text-slate-400 hover:text-rose-600 transition-colors">
            <i data-lucide="thumbs-down" class="w-3 h-3"></i>
          </button>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });

  if (window.lucide) lucide.createIcons();
}

// 4. Interactive Feedback Submission
async function submitFeedback(itemId, isPositive, btnEl) {
  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: itemId, is_positive: isPositive, trigger_rebalance: true })
    });
    const data = await res.json();
    btnEl.classList.add(isPositive ? 'text-emerald-600' : 'text-rose-600');
    loadStatus();
  } catch (e) {
    console.error("Feedback error:", e);
  }
}

// 5. Spectral Reflectance Chart (Query vs Top-1)
function initSpectralChart() {
  const ctx = document.getElementById('spectralComparisonChart').getContext('2d');
  spectralChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: BAND_LABELS,
      datasets: [
        {
          label: 'Query Spectral Profile',
          data: [],
          borderColor: '#7C3AED',
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          borderWidth: 2.5,
          tension: 0.3,
          fill: true
        },
        {
          label: 'Top-1 Retrieved Profile',
          data: [],
          borderColor: '#C4B5FD',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [4, 4],
          tension: 0.3,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#475569', font: { size: 11, weight: '500' } } }
      },
      scales: {
        x: { ticks: { color: '#64748B', font: { size: 9 }, maxRotation: 45 }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' }, title: { display: true, text: 'Reflectance Ratio', color: '#64748B', font: { size: 10 } } }
      }
    }
  });
}

function updateSpectralChart(querySig, topSig, className) {
  if (!spectralChartInstance) return;
  spectralChartInstance.data.datasets[0].label = `Query: ${className}`;
  spectralChartInstance.data.datasets[0].data = querySig || [];
  spectralChartInstance.data.datasets[1].data = topSig || [];
  spectralChartInstance.update();
}

// 6. Tier Architecture Overview & Rebalance
async function loadTiersOverview() {
  try {
    const res = await fetch('/api/tiers');
    const data = await res.json();

    const total = data.counts.TOTAL || 1500;
    document.getElementById('tier-count-hot').innerText = data.counts.HOT;
    document.getElementById('tier-count-warm').innerText = data.counts.WARM;
    document.getElementById('tier-count-cold').innerText = data.counts.COLD;

    document.getElementById('tier-pct-hot').innerText = `${((data.counts.HOT / total) * 100).toFixed(1)}%`;
    document.getElementById('tier-pct-warm').innerText = `${((data.counts.WARM / total) * 100).toFixed(1)}%`;
    document.getElementById('tier-pct-cold').innerText = `${((data.counts.COLD / total) * 100).toFixed(1)}%`;

    renderClassTierChart(data.class_tier_distribution);

    const strip = document.getElementById('top-hot-strip');
    strip.innerHTML = '';
    data.top_hot_items.forEach(item => {
      const img = document.createElement('img');
      img.src = item.preview_url;
      img.alt = item.id;
      img.title = `${item.class_name} (Act: ${item.activity_score})`;
      img.className = 'w-10 h-10 rounded-xl object-cover border border-amber-300 flex-shrink-0 shadow-sm';
      strip.appendChild(img);
    });
  } catch (e) {
    console.error("Tiers overview error:", e);
  }
}

function renderClassTierChart(matrix) {
  const ctx = document.getElementById('classTierChart').getContext('2d');
  const classes = Object.keys(matrix);
  const hotData = classes.map(c => matrix[c].HOT);
  const warmData = classes.map(c => matrix[c].WARM);
  const coldData = classes.map(c => matrix[c].COLD);

  if (classTierChartInstance) classTierChartInstance.destroy();

  classTierChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: classes,
      datasets: [
        { label: 'HOT', data: hotData, backgroundColor: '#F59E0B' },
        { label: 'WARM', data: warmData, backgroundColor: '#8B5CF6' },
        { label: 'COLD', data: coldData, backgroundColor: '#C4B5FD' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, ticks: { color: '#64748B', font: { size: 9 }, maxRotation: 45 }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { stacked: true, ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } }
      },
      plugins: {
        legend: { labels: { color: '#475569', font: { size: 11 } } }
      }
    }
  });
}

async function triggerTierRebalance() {
  const btn = document.getElementById('btn-rebalance');
  btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i> Reorganizing Tiers...';
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/rebalance', { method: 'POST' });
    const data = await res.json();

    const summaryBox = document.getElementById('rebalance-summary-box');
    summaryBox.classList.remove('hidden');
    document.getElementById('reb-promotions').innerText = `+${data.promotions} items promoted`;
    document.getElementById('reb-demotions').innerText = `-${data.demotions} items demoted`;
    document.getElementById('reb-time').innerText = `${data.duration_ms.toFixed(2)} ms`;

    await loadTiersOverview();
    await loadStatus();
  } catch (e) {
    console.error("Rebalance error:", e);
  } finally {
    btn.innerHTML = '<i data-lucide="repeat" class="w-4 h-4"></i> Trigger Dynamic Reorganization';
    if (window.lucide) lucide.createIcons();
  }
}

// 7. Empirical Benchmarks & Trade-Off Sweep
async function loadBenchmarkData() {
  try {
    const [resBench, resSweep] = await Promise.all([
      fetch('/api/benchmark'),
      fetch('/api/tradeoff_sweep')
    ]);

    const data = await resBench.json();
    const metrics = data.benchmark.metrics;
    const sweepData = await resSweep.json();

    // 1. Render Main Benchmark Table
    const tbody = document.getElementById('benchmark-table-body');
    tbody.innerHTML = `
      <tr class="hover:bg-purple-50/50 transition-colors">
        <td class="py-2.5 px-4 font-bold text-slate-700">Exhaustive Flat Baseline</td>
        <td class="py-2.5 px-4">${metrics.flat.precision_at_1}</td>
        <td class="py-2.5 px-4">${metrics.flat.precision_at_5}</td>
        <td class="py-2.5 px-4">${metrics.flat.precision_at_10}</td>
        <td class="py-2.5 px-4 text-purple-700 font-bold">${metrics.flat.mAP}</td>
        <td class="py-2.5 px-4">${metrics.flat.avg_search_time_ms} ms</td>
        <td class="py-2.5 px-4">${metrics.flat.avg_candidates_evaluated}</td>
        <td class="py-2.5 px-4 text-slate-400">0.0%</td>
        <td class="py-2.5 px-4 text-slate-600">1.00x</td>
      </tr>
      <tr class="hover:bg-purple-50/50 transition-colors">
        <td class="py-2.5 px-4 font-bold text-purple-800">Static Inverted Index (IVF)</td>
        <td class="py-2.5 px-4">${metrics.inverted.precision_at_1}</td>
        <td class="py-2.5 px-4">${metrics.inverted.precision_at_5}</td>
        <td class="py-2.5 px-4">${metrics.inverted.precision_at_10}</td>
        <td class="py-2.5 px-4 text-purple-700 font-bold">${metrics.inverted.mAP}</td>
        <td class="py-2.5 px-4">${metrics.inverted.avg_search_time_ms} ms</td>
        <td class="py-2.5 px-4">${metrics.inverted.avg_candidates_evaluated}</td>
        <td class="py-2.5 px-4 text-purple-600 font-bold">${metrics.inverted.search_space_reduction_pct}%</td>
        <td class="py-2.5 px-4 text-purple-800 font-semibold">${metrics.inverted.speedup_factor}x</td>
      </tr>
      <tr class="hover:bg-purple-100/50 bg-purple-50/70 border-l-4 border-purple-600 transition-colors">
        <td class="py-2.5 px-4 font-bold text-purple-900">Proposed Adaptive Multi-Tier</td>
        <td class="py-2.5 px-4 font-semibold text-slate-800">${metrics.adaptive.precision_at_1}</td>
        <td class="py-2.5 px-4 font-semibold text-slate-800">${metrics.adaptive.precision_at_5}</td>
        <td class="py-2.5 px-4 font-semibold text-slate-800">${metrics.adaptive.precision_at_10}</td>
        <td class="py-2.5 px-4 text-purple-700 font-bold">${metrics.adaptive.mAP}</td>
        <td class="py-2.5 px-4 text-purple-700 font-bold">${metrics.adaptive.avg_search_time_ms} ms</td>
        <td class="py-2.5 px-4 text-amber-600 font-bold">${metrics.adaptive.avg_candidates_evaluated}</td>
        <td class="py-2.5 px-4 text-purple-700 font-black text-sm">${metrics.adaptive.search_space_reduction_pct}%</td>
        <td class="py-2.5 px-4 text-purple-800 font-bold">${metrics.adaptive.speedup_factor}x</td>
      </tr>
    `;

    renderBenchmarkCharts(metrics);
    renderAblationList(data.ablation.ablations);

    // 2. Render Trade-Off Sweep Tables
    renderTradeoffTables(sweepData);

    // 3. Render Pareto & Scalability Charts
    renderParetoChart(sweepData);
    renderScalabilityChart(sweepData);

  } catch (e) {
    console.error("Benchmark load error:", e);
  }
}

function renderTradeoffTables(sweepData) {
  // Adaptive Sweep Table
  const adTbody = document.getElementById('adaptive-sweep-tbody');
  if (adTbody && sweepData.adaptive_sweep) {
    adTbody.innerHTML = sweepData.adaptive_sweep.map(row => `
      <tr class="hover:bg-purple-50/60 transition-colors ${row.threshold === 0.82 ? 'bg-purple-100/40 font-bold' : ''}">
        <td class="py-2 px-3 text-purple-900 font-bold">τ = ${row.threshold.toFixed(2)}</td>
        <td class="py-2 px-3 text-slate-700 font-sans">${row.policy} (${row.tier_behavior})</td>
        <td class="py-2 px-3 text-purple-700 font-bold">${row.reduction_pct.toFixed(1)}%</td>
        <td class="py-2 px-3 text-emerald-700 font-bold">${(row.p10 * 10).toFixed(1)}/10</td>
        <td class="py-2 px-3 text-slate-500">${row.latency_ms.toFixed(3)} ms</td>
      </tr>
    `).join('');
  }

  // IVF Sweep Table
  const ivfTbody = document.getElementById('ivf-sweep-tbody');
  if (ivfTbody && sweepData.ivf_sweep) {
    ivfTbody.innerHTML = sweepData.ivf_sweep.map(row => `
      <tr class="hover:bg-purple-50/60 transition-colors ${row.probes === 3 ? 'bg-purple-100/40 font-bold' : ''}">
        <td class="py-2 px-3 text-purple-900 font-bold">${row.probes} / 16</td>
        <td class="py-2 px-3 text-slate-700">${row.scanned} / 1500</td>
        <td class="py-2 px-3 text-purple-700 font-bold">${row.reduction_pct.toFixed(1)}%</td>
        <td class="py-2 px-3 text-emerald-700 font-bold">${(row.p10 * 10).toFixed(1)}/10</td>
        <td class="py-2 px-3 text-slate-500">${row.latency_ms.toFixed(3)} ms</td>
      </tr>
    `).join('');
  }
}

function renderParetoChart(sweepData) {
  const ctx = document.getElementById('paretoTradeoffChart');
  if (!ctx || !sweepData) return;

  if (paretoChartInstance) paretoChartInstance.destroy();

  const adPoints = sweepData.adaptive_sweep.map(s => ({
    x: s.reduction_pct,
    y: s.p10 * 10,
    label: `τ=${s.threshold.toFixed(2)} (${s.policy})`
  }));

  const ivfPoints = sweepData.ivf_sweep.map(s => ({
    x: s.reduction_pct,
    y: s.p10 * 10,
    label: `IVF ${s.probes} probes`
  }));

  const flatPoint = [{
    x: 0,
    y: sweepData.baseline_flat.p10 * 10,
    label: 'Exhaustive Flat Ground Truth'
  }];

  paretoChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Proposed Adaptive AI Multi-Tier',
          data: adPoints,
          backgroundColor: '#7C3AED',
          borderColor: '#7C3AED',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: true,
          borderDash: [3, 3]
        },
        {
          label: 'Static Inverted Index (IVF)',
          data: ivfPoints,
          backgroundColor: '#3B82F6',
          borderColor: '#3B82F6',
          pointRadius: 5,
          showLine: true,
          borderDash: [2, 2]
        },
        {
          label: 'Exhaustive Flat Baseline',
          data: flatPoint,
          backgroundColor: '#10B981',
          pointRadius: 8,
          pointStyle: 'rectRot'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Search-Space Reduction (%)', color: '#64748B', font: { size: 10 } },
          ticks: { color: '#64748B', font: { size: 9 } },
          grid: { color: 'rgba(124, 58, 237, 0.06)' },
          min: 0,
          max: 100
        },
        y: {
          title: { display: true, text: 'Top-10 Accuracy (Matches / 10)', color: '#64748B', font: { size: 10 } },
          ticks: { color: '#64748B', font: { size: 9 } },
          grid: { color: 'rgba(124, 58, 237, 0.06)' },
          min: 3,
          max: 7
        }
      },
      plugins: {
        legend: { labels: { color: '#475569', font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: function(context) {
              const p = context.raw;
              return `${p.label || context.dataset.label}: ${p.x.toFixed(1)}% reduction, ${p.y.toFixed(2)}/10 accuracy`;
            }
          }
        }
      }
    }
  });
}

function renderScalabilityChart(sweepData) {
  const ctx = document.getElementById('scalabilityChart');
  if (!ctx || !sweepData || !sweepData.scalability_crossover) return;

  if (scalabilityChartInstance) scalabilityChartInstance.destroy();

  const labels = sweepData.scalability_crossover.map(s => `${(s.dataset_size/1000).toFixed(1)}k`);
  const flatLat = sweepData.scalability_crossover.map(s => s.flat_latency_ms);
  const ivfLat = sweepData.scalability_crossover.map(s => s.ivf_latency_ms);
  const adLat = sweepData.scalability_crossover.map(s => s.adaptive_latency_ms);

  scalabilityChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Exhaustive Flat Scan O(N)',
          data: flatLat,
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          borderWidth: 2,
          tension: 0.2
        },
        {
          label: 'Static Inverted IVF O(n_probe * N/K)',
          data: ivfLat,
          borderColor: '#3B82F6',
          borderWidth: 2,
          tension: 0.2
        },
        {
          label: 'Proposed Adaptive Multi-Tier O(N_HOT)',
          data: adLat,
          borderColor: '#7C3AED',
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          fill: true,
          borderWidth: 2.5,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Repository Scale (Images)', color: '#64748B', font: { size: 10 } },
          ticks: { color: '#64748B', font: { size: 9 } },
          grid: { color: 'rgba(124, 58, 237, 0.06)' }
        },
        y: {
          title: { display: true, text: 'Retrieval Latency (ms)', color: '#64748B', font: { size: 10 } },
          ticks: { color: '#64748B', font: { size: 9 } },
          grid: { color: 'rgba(124, 58, 237, 0.06)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#475569', font: { size: 10 } } }
      }
    }
  });
}

function renderBenchmarkCharts(metrics) {
  const ctx1 = document.getElementById('benchmarkPrecisionChart').getContext('2d');
  if (benchPrecisionChartInstance) benchPrecisionChartInstance.destroy();

  benchPrecisionChartInstance = new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: ['Precision@1', 'Precision@5', 'Precision@10', 'mAP'],
      datasets: [
        { label: 'Exhaustive Flat', data: [metrics.flat.precision_at_1, metrics.flat.precision_at_5, metrics.flat.precision_at_10, metrics.flat.mAP], backgroundColor: '#94A3B8' },
        { label: 'Static Inverted', data: [metrics.inverted.precision_at_1, metrics.inverted.precision_at_5, metrics.inverted.precision_at_10, metrics.inverted.mAP], backgroundColor: '#C4B5FD' },
        { label: 'Adaptive Multi-Tier', data: [metrics.adaptive.precision_at_1, metrics.adaptive.precision_at_5, metrics.adaptive.precision_at_10, metrics.adaptive.mAP], backgroundColor: '#7C3AED' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#64748B', font: { size: 10 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { min: 0.0, max: 1.0, ticks: { color: '#64748B', font: { size: 10 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } }
      },
      plugins: { legend: { labels: { color: '#475569', font: { size: 10 } } } }
    }
  });

  const ctx2 = document.getElementById('benchmarkEfficiencyChart').getContext('2d');
  if (benchEfficiencyChartInstance) benchEfficiencyChartInstance.destroy();

  benchEfficiencyChartInstance = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: ['Flat Exhaustive', 'Static Inverted', 'Adaptive Proposed'],
      datasets: [
        {
          label: 'Search-Space Reduction (%)',
          data: [0, metrics.inverted.search_space_reduction_pct, metrics.adaptive.search_space_reduction_pct],
          backgroundColor: '#7C3AED',
          yAxisID: 'y'
        },
        {
          label: 'Candidates Evaluated',
          data: [metrics.flat.avg_candidates_evaluated, metrics.inverted.avg_candidates_evaluated, metrics.adaptive.avg_candidates_evaluated],
          backgroundColor: '#DDD6FE',
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#64748B', font: { size: 10 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { type: 'linear', position: 'left', min: 0, max: 100, ticks: { color: '#7C3AED' }, grid: { color: 'rgba(124, 58, 237, 0.06)' }, title: { display: true, text: 'Reduction %', color: '#7C3AED' } },
        y1: { type: 'linear', position: 'right', min: 0, ticks: { color: '#8B5CF6' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Candidates', color: '#8B5CF6' } }
      },
      plugins: { legend: { labels: { color: '#475569', font: { size: 10 } } } }
    }
  });
}

function renderAblationList(ablations) {
  const container = document.getElementById('ablation-list');
  container.innerHTML = '';
  ablations.forEach(ab => {
    const row = document.createElement('div');
    row.className = 'p-3 bg-purple-50/50 rounded-xl border border-purple-100 flex items-center justify-between text-xs hover:bg-purple-50 transition-colors';
    row.innerHTML = `
      <div>
        <span class="font-bold text-slate-800 block">${ab.name}</span>
        <span class="text-[11px] text-slate-500">Search Space Reduction: <strong class="text-purple-700 font-bold">${ab.search_space_reduction_pct}%</strong></span>
      </div>
      <div class="text-right font-mono">
        <span class="text-purple-700 font-bold block">mAP: ${ab.mAP}</span>
        <span class="text-[10px] text-slate-400">${ab.avg_search_time_ms} ms</span>
      </div>
    `;
    container.appendChild(row);
  });

  const ctx = document.getElementById('ablationChart').getContext('2d');
  if (ablationChartInstance) ablationChartInstance.destroy();

  ablationChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ablations.map(a => a.name.split('(')[0].trim()),
      datasets: [
        { label: 'mAP', data: ablations.map(a => a.mAP), backgroundColor: '#7C3AED' },
        { label: 'Precision@1', data: ablations.map(a => a.precision_at_1), backgroundColor: '#C4B5FD' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#64748B', font: { size: 9 }, maxRotation: 30 }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { min: 0.4, max: 0.9, ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } }
      },
      plugins: { legend: { labels: { color: '#475569', font: { size: 10 } } } }
    }
  });
}

// 8. 13-Band Spectral Signatures Explorer
async function loadSpectralSignatures() {
  try {
    const res = await fetch('/api/spectral_signatures');
    const data = await res.json();
    const ctx = document.getElementById('spectralSignaturesChart').getContext('2d');

    const colors = [
      '#7C3AED', '#8B5CF6', '#A855F7', '#6366F1', '#3B82F6',
      '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#64748B'
    ];

    const datasets = Object.keys(data.profiles).map((c, i) => ({
      label: c,
      data: data.profiles[c],
      borderColor: colors[i % colors.length],
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 2,
      fill: false
    }));

    if (spectralSignaturesChartInstance) spectralSignaturesChartInstance.destroy();

    spectralSignaturesChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.bands,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#64748B', font: { size: 8 }, maxRotation: 45 }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
          y: { ticks: { color: '#64748B', font: { size: 8 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' }, title: { display: true, text: 'Mean Reflectance', color: '#64748B', font: { size: 9 } } }
        },
        plugins: {
          legend: { position: 'bottom', labels: { color: '#475569', boxWidth: 10, font: { size: 9 } } }
        }
      }
    });
  } catch (e) {
    console.error("Signatures error:", e);
  }
}

// 9. Workload Simulation
async function runLiveSimulation() {
  const scenario = document.getElementById('sim-scenario').value;
  const numQueries = parseInt(document.getElementById('sim-queries').value);
  const interval = parseInt(document.getElementById('sim-interval').value);
  const btn = document.getElementById('btn-run-sim');

  btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin text-white"></i> Simulating Activity...';
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: scenario, num_queries: numQueries, rebalance_interval: interval })
    });
    const data = await res.json();
    renderSimulationResults(data);
    await loadStatus();
  } catch (e) {
    console.error("Sim error:", e);
  } finally {
    btn.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i> Run Workload Simulation';
    if (window.lucide) lucide.createIcons();
  }
}

function renderSimulationResults(data) {
  const ctx = document.getElementById('simProgressionChart').getContext('2d');
  const labels = data.tier_progression.map((_, i) => i === 0 ? 'Initial' : `Interval ${i}`);
  const hotData = data.tier_progression.map(t => t.HOT);
  const warmData = data.tier_progression.map(t => t.WARM);
  const coldData = data.tier_progression.map(t => t.COLD);

  if (simProgressionChartInstance) simProgressionChartInstance.destroy();

  simProgressionChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'HOT', data: hotData, borderColor: '#F59E0B', backgroundColor: '#F59E0B', tension: 0.2 },
        { label: 'WARM', data: warmData, borderColor: '#8B5CF6', backgroundColor: '#8B5CF6', tension: 0.2 },
        { label: 'COLD', data: coldData, borderColor: '#C4B5FD', backgroundColor: '#C4B5FD', tension: 0.2 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } },
        y: { ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(124, 58, 237, 0.06)' } }
      },
      plugins: { legend: { labels: { color: '#475569', font: { size: 10 } } } }
    }
  });

  const ctx2 = document.getElementById('simHotClassChart').getContext('2d');
  const classes = Object.keys(data.hot_class_distribution);
  const counts = Object.values(data.hot_class_distribution);

  if (simHotClassChartInstance) simHotClassChartInstance.destroy();

  simHotClassChartInstance = new Chart(ctx2, {
    type: 'doughnut',
    data: {
      labels: classes,
      datasets: [{
        data: counts,
        backgroundColor: ['#7C3AED', '#8B5CF6', '#A855F7', '#6366F1', '#3B82F6', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#C4B5FD']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#475569', font: { size: 9 } } }
      }
    }
  });

  const logContainer = document.getElementById('sim-step-logs');
  logContainer.innerHTML = '';
  data.reorganization_steps.forEach(step => {
    const item = document.createElement('div');
    item.className = 'p-2 bg-purple-50/60 rounded-xl border border-purple-100 flex justify-between text-xs';
    item.innerHTML = `
      <span class="text-slate-700">Step ${step.step} (${step.query_class}):</span>
      <span class="text-purple-700 font-semibold">+${step.promotions} Promoted</span>
      <span class="text-rose-500 font-semibold">-${step.demotions} Demoted</span>
      <span class="text-amber-600 font-bold">HOT: ${step.tier_counts.HOT}</span>
    `;
    logContainer.appendChild(item);
  });
}
