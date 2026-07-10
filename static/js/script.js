// =====================================================================
// CardioSense — frontend logic
// =====================================================================
const FIELD_LABELS = {
  age: "Age", sex: "Sex", cp: "Chest pain type", trestbps: "Resting BP (mm Hg)",
  chol: "Cholesterol (mg/dl)", fbs: "Fasting blood sugar > 120", restecg: "Resting ECG",
  thalach: "Max heart rate", exang: "Exercise-induced angina", oldpeak: "ST depression",
  slope: "ST slope", ca: "Major vessels", thal: "Thalassemia",
};

const LOOKUP = {
  sex: { 1: "Male", 0: "Female" },
  cp: { 0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic" },
  fbs: { 1: "Yes", 0: "No" },
  restecg: { 0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy" },
  exang: { 1: "Yes", 0: "No" },
  slope: { 0: "Upsloping", 1: "Flat", 2: "Downsloping" },
  thal: { 1: "Normal", 2: "Fixed defect", 3: "Reversible defect" },
};

let pieChart = null;
let barChart = null;
let lastResult = null;

// ---------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "dashboard") loadStats();
    if (btn.dataset.view === "history") loadHistory();
  });
});

// ---------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------
const themeToggle = document.getElementById("theme-toggle");
const iconSun = document.getElementById("icon-sun");
const iconMoon = document.getElementById("icon-moon");
const themeLabel = document.getElementById("theme-label");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.body.setAttribute("data-theme", theme);
  if (theme === "dark") {
    iconSun.style.display = "none";
    iconMoon.style.display = "inline";
    themeLabel.textContent = "Light mode";
  } else {
    iconSun.style.display = "inline";
    iconMoon.style.display = "none";
    themeLabel.textContent = "Dark mode";
  }
}
let currentTheme = "light";
themeToggle.addEventListener("click", () => {
  currentTheme = currentTheme === "light" ? "dark" : "light";
  applyTheme(currentTheme);
});
applyTheme(currentTheme);

// ---------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2600);
}

// ---------------------------------------------------------------------
// Form validation
// ---------------------------------------------------------------------
const form = document.getElementById("predict-form");
const requiredFields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal"];

function validateField(name) {
  const el = document.getElementById(name);
  const errorEl = el.closest(".field").querySelector(".error-msg");
  const value = el.value.trim();
  if (value === "") {
    el.classList.add("invalid");
    errorEl.textContent = "Required";
    return false;
  }
  if (el.tagName === "INPUT" && el.type === "number") {
    const num = parseFloat(value);
    const min = parseFloat(el.min);
    const max = parseFloat(el.max);
    if (isNaN(num) || num < min || num > max) {
      el.classList.add("invalid");
      errorEl.textContent = `Must be ${min}–${max}`;
      return false;
    }
  }
  el.classList.remove("invalid");
  errorEl.textContent = "";
  return true;
}

requiredFields.forEach((name) => {
  const el = document.getElementById(name);
  el.addEventListener("blur", () => validateField(name));
  el.addEventListener("input", () => {
    if (el.classList.contains("invalid")) validateField(name);
  });
});

document.getElementById("reset-btn").addEventListener("click", () => {
  form.reset();
  requiredFields.forEach((name) => {
    const el = document.getElementById(name);
    el.classList.remove("invalid");
    el.closest(".field").querySelector(".error-msg").textContent = "";
  });
  showResultState("empty");
});

// ---------------------------------------------------------------------
// Result panel state switching
// ---------------------------------------------------------------------
function showResultState(state) {
  document.getElementById("loading-card").hidden = state !== "loading";
  document.getElementById("empty-card").hidden = state !== "empty";
  document.getElementById("result-card").hidden = state !== "result";
}

// ---------------------------------------------------------------------
// Submit prediction
// ---------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const allValid = requiredFields.map(validateField).every(Boolean);
  if (!allValid) {
    showToast("Please fix the highlighted fields");
    return;
  }

  const btn = document.getElementById("predict-btn");
  btn.querySelector(".btn-label").style.opacity = "0";
  btn.querySelector(".btn-spinner").hidden = false;
  btn.disabled = true;
  showResultState("loading");

  const payload = { patient_name: document.getElementById("patient_name").value.trim() || "Unnamed Patient" };
  requiredFields.forEach((name) => {
    payload[name] = document.getElementById(name).value;
  });

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    // small delay so the ECG loading animation is visible / feels analyzed
    await new Promise((r) => setTimeout(r, 900));

    if (!data.success) {
      showToast(data.errors ? data.errors[0] : "Prediction failed");
      showResultState("empty");
    } else {
      renderResult(data);
      showResultState("result");
      showToast("Prediction complete");
    }
  } catch (err) {
    showToast("Server error — is the Flask app running?");
    showResultState("empty");
  } finally {
    btn.querySelector(".btn-label").style.opacity = "1";
    btn.querySelector(".btn-spinner").hidden = true;
    btn.disabled = false;
  }
});

function renderResult(data) {
  lastResult = data;
  const isHigh = data.risk_label === "High Risk";

  const badge = document.getElementById("result-badge");
  badge.className = "result-badge " + (isHigh ? "high" : "low");
  document.getElementById("result-label").textContent = isHigh ? "⚠ High Risk" : "✓ Low Risk";

  document.getElementById("confidence-value").textContent = `${data.confidence}%`;
  const ring = document.getElementById("ring-fg");
  const circumference = 2 * Math.PI * 52;
  const offset = circumference - (data.confidence / 100) * circumference;
  ring.style.stroke = isHigh ? "var(--high)" : "var(--low)";
  requestAnimationFrame(() => { ring.style.strokeDashoffset = offset; });

  document.getElementById("result-timestamp").textContent = `Predicted on ${data.timestamp} · Model: ${data.model_used}`;
  document.getElementById("recommendation-text").textContent = data.recommendation;

  const grid = document.getElementById("summary-grid");
  grid.innerHTML = "";
  const rows = [
    ["Patient", data.patient_name],
    ...Object.entries(data.input).map(([k, v]) => {
      const label = FIELD_LABELS[k] || k;
      const displayVal = LOOKUP[k] ? LOOKUP[k][v] : v;
      return [label, displayVal];
    }),
  ];
  rows.forEach(([label, val]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span>${label}</span><span>${val}</span>`;
    grid.appendChild(div);
  });
}

// ---------------------------------------------------------------------
// PDF export
// ---------------------------------------------------------------------
document.getElementById("export-pdf-btn").addEventListener("click", () => {
  if (!lastResult) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  const isHigh = lastResult.risk_label === "High Risk";

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(13, 148, 136);
  doc.text("CardioSense — Heart Disease Risk Report", 14, 20);

  doc.setDrawColor(220, 220, 220);
  doc.line(14, 25, 196, 25);

  doc.setFontSize(11);
  doc.setTextColor(60, 60, 60);
  doc.setFont("helvetica", "normal");
  doc.text(`Patient: ${lastResult.patient_name}`, 14, 36);
  doc.text(`Date: ${lastResult.timestamp}`, 14, 43);
  doc.text(`Model used: ${lastResult.model_used}`, 14, 50);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(isHigh ? 225 : 5, isHigh ? 29 : 150, isHigh ? 72 : 105);
  doc.text(`Result: ${lastResult.risk_label}  (${lastResult.confidence}% confidence)`, 14, 62);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.setTextColor(30, 30, 30);
  doc.text("Recommendation", 14, 74);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  const recLines = doc.splitTextToSize(lastResult.recommendation, 182);
  doc.text(recLines, 14, 81);

  let y = 81 + recLines.length * 5.5 + 10;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text("Patient data", 14, y);
  y += 8;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  Object.entries(lastResult.input).forEach(([k, v]) => {
    const label = FIELD_LABELS[k] || k;
    const displayVal = LOOKUP[k] ? LOOKUP[k][v] : v;
    doc.text(`${label}: ${displayVal}`, 14, y);
    y += 6.5;
    if (y > 280) { doc.addPage(); y = 20; }
  });

  y += 6;
  doc.setFontSize(8.5);
  doc.setTextColor(140, 140, 140);
  doc.text("This report is generated by a machine learning screening tool and is not a substitute", 14, y);
  doc.text("for professional medical diagnosis. Please consult a licensed physician.", 14, y + 5);

  doc.save(`CardioSense_Report_${lastResult.patient_name.replace(/\s+/g, "_")}.pdf`);
});

// ---------------------------------------------------------------------
// Dashboard stats + charts
// ---------------------------------------------------------------------
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    if (!data.success) return;

    document.getElementById("stat-total").textContent = data.total_predictions;
    document.getElementById("stat-high").textContent = data.high_risk_count;
    document.getElementById("stat-low").textContent = data.low_risk_count;
    document.getElementById("stat-conf").textContent = `${data.average_confidence}%`;
    document.getElementById("best-model-name").textContent = data.best_model;

    renderPieChart(data.high_risk_count, data.low_risk_count);
    renderBarChart(data.model_accuracies);
  } catch (err) {
    console.error(err);
  }
}

function renderPieChart(high, low) {
  const ctx = document.getElementById("chart-pie");
  const styles = getComputedStyle(document.documentElement);
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["High risk", "Low risk"],
      datasets: [{
        data: [high, low],
        backgroundColor: [styles.getPropertyValue("--high").trim(), styles.getPropertyValue("--low").trim()],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: styles.getPropertyValue("--text").trim(), font: { family: "Inter" } } } },
      cutout: "68%",
    },
  });
}

function renderBarChart(accuracies) {
  const ctx = document.getElementById("chart-bar");
  const styles = getComputedStyle(document.documentElement);
  const labels = Object.keys(accuracies || {});
  const values = labels.map((l) => accuracies[l].accuracy);
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Accuracy (%)",
        data: values,
        backgroundColor: styles.getPropertyValue("--accent").trim(),
        borderRadius: 6,
        maxBarThickness: 40,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: styles.getPropertyValue("--muted").trim() }, grid: { color: styles.getPropertyValue("--border").trim() } },
        x: { ticks: { color: styles.getPropertyValue("--muted").trim(), font: { size: 10 } }, grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------
// History table + search
// ---------------------------------------------------------------------
async function loadHistory(query = "") {
  try {
    const url = query ? `/api/search?q=${encodeURIComponent(query)}` : "/api/history";
    const res = await fetch(url);
    const data = await res.json();
    const rows = data.history || data.results || [];
    const tbody = document.getElementById("history-body");
    const emptyEl = document.getElementById("empty-history");
    tbody.innerHTML = "";

    if (rows.length === 0) {
      emptyEl.hidden = false;
      return;
    }
    emptyEl.hidden = true;

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      const isHigh = r.risk_label === "High Risk";
      tr.innerHTML = `
        <td>${r.patient_name}</td>
        <td>${r.age}</td>
        <td>${r.sex == 1 ? "Male" : "Female"}</td>
        <td><span class="risk-pill ${isHigh ? "high" : "low"}">${r.risk_label}</span></td>
        <td>${r.confidence}%</td>
        <td>${r.created_at}</td>
        <td><button class="delete-btn" data-id="${r.id}">Delete</button></td>
      `;
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll(".delete-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await fetch(`/api/delete/${btn.dataset.id}`, { method: "DELETE" });
        loadHistory(document.getElementById("search-input").value.trim());
        showToast("Prediction deleted");
      });
    });
  } catch (err) {
    console.error(err);
  }
}

let searchTimer = null;
document.getElementById("search-input").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadHistory(e.target.value.trim()), 300);
});

// ---------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------
loadStats();
