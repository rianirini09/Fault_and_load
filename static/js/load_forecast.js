/**
 * load_forecast.js
 * -----------------
 * Alur sama seperti fault detection:
 *   isi form -> klik "Prediksi Beban" -> POST ke /api/load-forecast/predict
 *   -> hasil (kartu ringkasan + grafik 24 jam + status) muncul di kartu kanan.
 */

const WEATHER_FIELDS = ["T2M", "QV2M", "TQL", "W2M"];

let chartInstance = null;

/* ---------- Isi default "waktu sekarang" (dibulatkan ke jam) ---------- */
function setDefaultDatetime() {
  const el = document.getElementById("target_datetime");
  if (!el || el.value) return;
  const d = new Date();
  d.setMinutes(0, 0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  el.value = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:00`;
}

/* ---------- Ambil nilai form -> payload untuk backend ---------- */
function buildPayload() {
  const num = (id) => document.getElementById(id).value;

  const weather = {};
  WEATHER_FIELDS.forEach((f) => { weather[f] = num(f); });

  return {
    target_datetime: num("target_datetime"),
    current_demand: num("current_demand"),
    demand_1h_ago: num("demand_1h_ago"),
    demand_2h_ago: num("demand_2h_ago"),
    demand_24h_ago: num("demand_24h_ago"),
    demand_168h_ago: num("demand_168h_ago"),
    roll_mean_24: num("roll_mean_24") || null,
    capacity_mw: num("capacity_mw") ? Number(num("capacity_mw")) : null,
    weather,
    calendar: {
      Holiday_ID: Number(num("holiday_id") || 0),
      holiday: document.getElementById("is_holiday").checked ? 1 : 0,
      school: document.getElementById("is_school").checked ? 1 : 0,
    },
  };
}

/* ---------- Submit ---------- */
document.getElementById("loadForecastForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const submitBtn = document.getElementById("submitBtn");
  const loading = document.getElementById("loading");
  const errorBox = document.getElementById("lf-error");
  const results = document.getElementById("lf-results");

  errorBox.classList.add("hidden");
  results.classList.add("hidden");
  submitBtn.disabled = true;
  loading.classList.remove("hidden");

  try {
    const res = await fetch("/api/load-forecast/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Terjadi kesalahan pada server.");

    // tampilkan dulu container-nya supaya Chart.js bisa mengukur ukuran canvas
    results.classList.remove("hidden");
    renderResults(data.result);
  } catch (err) {
    errorBox.textContent = "❌ " + err.message;
    errorBox.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    loading.classList.add("hidden");
  }
});

document.getElementById("loadForecastForm").addEventListener("reset", () => {
  document.getElementById("lf-results").classList.add("hidden");
  document.getElementById("lf-error").classList.add("hidden");
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  setTimeout(setDefaultDatetime, 0);
});

/* ---------- Render hasil ---------- */
function renderResults(r) {
  fillStatCards(r);
  fillStatusBadge(r);
  renderChart(r);
}

function fillStatCards(r) {
  document.getElementById("stat-current").textContent = `${formatNumber(r.current_demand_mw)} MW`;
  document.getElementById("stat-peak").textContent = `${formatNumber(r.predicted_peak_mw)} MW`;
  document.getElementById("stat-peak-hour").textContent = r.peak_hour;
  document.getElementById("stat-mape").textContent = `${r.mape_pct}%`;
}

function fillStatusBadge(r) {
  const badge = document.getElementById("status-badge");
  const detail = document.getElementById("status-detail");

  badge.textContent = `Status: ${r.status.label}`;
  badge.className = "lf-badge " + {
    ok: "lf-badge--ok",
    warning: "lf-badge--warning",
    critical: "lf-badge--critical",
  }[r.status.level];

  detail.textContent =
    `Rasio puncak ${r.status.ratio_pct}% dari kapasitas (${formatNumber(r.capacity_mw)} MW)` +
    (r.status.level !== "ok" ? ", siapkan cadangan" : "");
}

function formatNumber(n) {
  return new Intl.NumberFormat("id-ID", { maximumFractionDigits: 0 }).format(n);
}

function formatHour(iso) {
  return new Date(iso).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });
}

function renderChart(r) {
  const ctx = document.getElementById("loadForecastChart").getContext("2d");

  const historyLabels = r.actual_history.map((p) => formatHour(p.datetime));
  const forecastLabels = r.forecast.map((p) => formatHour(p.datetime));
  const labels = [...historyLabels, ...forecastLabels];
  const nowIndex = historyLabels.length - 1;

  const actualSeries = [
    ...r.actual_history.map((p) => p.value),
    ...new Array(r.forecast.length).fill(null),
  ];

  // garis prediksi nyambung dari titik "sekarang"
  const predSeries = [
    ...new Array(historyLabels.length - 1).fill(null),
    r.actual_history[r.actual_history.length - 1].value,
    ...r.forecast.map((p) => p.q50),
  ];

  const q10Series = [...new Array(historyLabels.length).fill(null), ...r.forecast.map((p) => p.q10)];
  const q90Series = [...new Array(historyLabels.length).fill(null), ...r.forecast.map((p) => p.q90)];

  const chartData = {
    labels,
    datasets: [
      { label: "Batas atas interval", data: q90Series, borderWidth: 0, pointRadius: 0,
        fill: "+1", backgroundColor: "rgba(33, 203, 243, 0.16)", order: 3 },
      { label: "Batas bawah interval", data: q10Series, borderWidth: 0, pointRadius: 0,
        fill: false, order: 3 },
      { label: "Prediksi", data: predSeries, borderColor: "#21cbf3", borderDash: [6, 4],
        borderWidth: 2, pointRadius: 0, tension: 0.35, order: 1 },
      { label: "Aktual", data: actualSeries, borderColor: "#e0e0e0", borderWidth: 2,
        pointRadius: 3, pointBackgroundColor: "#e0e0e0", tension: 0.35, order: 1 },
    ],
  };

  const options = {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (item) => (item.raw === null ? null : `${item.dataset.label}: ${formatNumber(item.raw)} MW`),
        },
      },
      annotation: {
        annotations: {
          capacityLine: {
            type: "line", yMin: r.capacity_mw, yMax: r.capacity_mw,
            borderColor: "#ff8a80", borderDash: [4, 4], borderWidth: 1.5,
            label: {
              display: true, content: "Kapasitas", position: "end",
              xAdjust: -6, yAdjust: -14,
              color: "#ff8a80", backgroundColor: "rgba(244, 67, 54, 0.16)",
              padding: { x: 6, y: 3 }, borderRadius: 4,
              font: { weight: "600", size: 11 },
            },
          },
          nowLine: {
            type: "line", xMin: nowIndex, xMax: nowIndex,
            borderColor: "#8ecae6", borderDash: [3, 3], borderWidth: 1,
            label: {
              display: true, content: "Sekarang", position: "start",
              yAdjust: 8, xAdjust: 6, rotation: -90,
              color: "#8ecae6", backgroundColor: "rgba(142, 202, 230, 0.14)",
              padding: { x: 6, y: 3 }, borderRadius: 4,
              font: { weight: "500", size: 11 },
            },
          },
        },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: "#b0b0b0", maxTicksLimit: 10 } },
      y: {
        title: { display: true, text: "MW", color: "#b0b0b0" },
        ticks: { color: "#b0b0b0" },
        grid: { color: "rgba(255, 255, 255, 0.08)" },
      },
    },
  };

  if (chartInstance) chartInstance.destroy();
  chartInstance = new Chart(ctx, { type: "line", data: chartData, options });
}

document.addEventListener("DOMContentLoaded", setDefaultDatetime);