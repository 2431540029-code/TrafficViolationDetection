const catalog = JSON.parse(
  document.querySelector("#model-catalog").textContent,
);
const taskSelect = document.querySelector("#task-select");
const splitSelect = document.querySelector("#split-select");
const confidenceInput = document.querySelector("#confidence-input");
const confidenceValue = document.querySelector("#confidence-value");
const repeatsInput = document.querySelector("#repeats-input");
const modelOptions = document.querySelector("#model-options");
const modelCount = document.querySelector("#model-count");
const modelAvailabilityNote = document.querySelector(
  "#model-availability-note",
);
const datasetInfo = document.querySelector("#selected-dataset-info");
const runButton = document.querySelector("#run-comparison");
const loading = document.querySelector("#comparison-loading");
const errorBox = document.querySelector("#comparison-error");
const resultsContext = document.querySelector("#results-context");
const summary = document.querySelector("#benchmark-summary");
const resultsBody = document.querySelector("#results-table-body");
const commentary = document.querySelector("#results-commentary");
const historyBody = document.querySelector("#history-table-body");
const colors = ["#5de0b0", "#8bb4ff", "#ffcb83"];

function escapeText(value) {
  return String(value ?? "");
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Chưa đo";
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "Chưa đo";
}

function selectedTaskModels() {
  return catalog.filter((model) => model.task === taskSelect.value);
}

function currentSplitInfo() {
  const firstModel = selectedTaskModels()[0];
  return firstModel?.splits?.[splitSelect.value] ?? null;
}

function renderModelOptions() {
  const models = selectedTaskModels();
  modelOptions.replaceChildren();

  if (!models.length) {
    modelCount.textContent = "0 checkpoint sẵn sàng";
    modelOptions.append(
      Object.assign(document.createElement("p"), {
        className: "history-empty",
        textContent: "Chưa cấu hình mô hình cho bài toán này.",
      }),
    );
    return;
  }

  const availableCount = models.filter((model) => model.available).length;
  const incompleteCount = models.filter(
    (model) => model.training_incomplete,
  ).length;
  modelCount.textContent = `${availableCount} / ${models.length} checkpoint sẵn sàng`;
  for (const model of models) {
    const label = document.createElement("label");
    label.className = "model-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "model-id";
    checkbox.value = model.id;
    checkbox.checked = model.available;
    checkbox.disabled = !model.available;

    const copy = document.createElement("span");
    copy.className = "model-option-copy";
    const title = document.createElement("strong");
    title.textContent = model.name;
    const details = document.createElement("small");
    details.textContent =
      `${model.architecture} · ${model.classes.length} lớp (${model.classes.join(", ")}) · ` +
      `640 px · ngưỡng web ${Math.round(model.system_confidence * 100)}% · ${model.weight_file} · ` +
      `${model.model_size_mb === null ? (model.training_incomplete ? "Có checkpoint dở dang; tiếp tục train để bật so sánh" : "Chưa fine-tune trên bộ dữ liệu này") : `${model.model_size_mb.toFixed(2)} MB`}`;
    copy.append(title, details);

    const status = document.createElement("span");
    status.className = `table-status ${model.available ? "status-active" : "status-trial"}`;
    status.textContent = model.available
      ? "SẴN SÀNG"
      : model.training_incomplete
        ? "CHƯA HOÀN TẤT"
        : "CHƯA HUẤN LUYỆN";
    label.append(checkbox, copy, status);
    modelOptions.append(label);
  }

  if (availableCount === 0) {
    modelAvailabilityNote.textContent =
      incompleteCount > 0
        ? "Có lượt fine-tune đang dở dang; model chỉ bật để so sánh sau khi huấn luyện hoàn tất."
        : "Chưa có checkpoint khả dụng cho bài toán này.";
  } else if (availableCount === 1) {
    modelAvailabilityNote.textContent =
      "Các kiến trúc chỉ so sánh được sau khi fine-tune trên cùng bộ dữ liệu và lớp. Hãy chạy lệnh huấn luyện được ghi trong README.";
  } else if (availableCount < models.length) {
    modelAvailabilityNote.textContent =
      `${availableCount} mô hình đã fine-tune; các kiến trúc còn lại cần huấn luyện trước khi so sánh. Dùng cùng tập đánh giá để đối chiếu.`;
  } else {
    modelAvailabilityNote.textContent =
      "Chọn tối đa ba checkpoint đã fine-tune cùng bài toán; các phép đo dùng chung tập, kích thước ảnh và thiết bị.";
  }
  for (const option of splitSelect.options) {
    const splitInfo = models[0].splits?.[option.value];
    option.disabled = !splitInfo?.available;
  }
  if (splitSelect.selectedOptions[0]?.disabled) {
    const availableSplit = [...splitSelect.options].find(
      (option) => !option.disabled,
    );
    if (availableSplit) splitSelect.value = availableSplit.value;
  }
  updateDatasetInfo();
}

function updateDatasetInfo() {
  const info = currentSplitInfo();
  const splitLabel = splitSelect.value === "test" ? "test độc lập" : "validation";
  datasetInfo.replaceChildren();
  if (!info || !info.available) {
    datasetInfo.textContent = info?.error
      ? `Không thể dùng tập ${splitLabel}: ${info.error}`
      : `Không tìm thấy ảnh hoặc nhãn hợp lệ cho tập ${splitLabel}.`;
    runButton.disabled = true;
    return;
  }
  const firstModel = selectedTaskModels()[0];
  const strong = document.createElement("strong");
  strong.textContent = `${firstModel.dataset_label} · ${splitLabel}`;
  const description = document.createElement("span");
  description.textContent =
    ` — ${info.image_count} ảnh, ${info.object_count} đối tượng có nhãn.` +
    (splitSelect.value === "val"
      ? " Đây là validation, không phải test độc lập."
      : " Hãy kiểm tra chất lượng nhãn test trước khi diễn giải.");
  if (info.orphan_label_count > 0) {
    description.textContent +=
      ` Có ${info.orphan_label_count} file nhãn không có ảnh tương ứng; các file này không được tính.`;
  }
  datasetInfo.append(strong, description);
  runButton.disabled = !selectedTaskModels().some((model) => model.available);
}

function makeSvgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function renderBarChart(containerId, rows, series, options = {}) {
  const container = document.querySelector(`#${containerId}`);
  container.replaceChildren();
  const measuredRows = rows.filter((row) =>
    series.some((item) => Number.isFinite(row[item.key])),
  );
  if (!measuredRows.length) {
    container.append(
      Object.assign(document.createElement("div"), {
        className: "chart-empty",
        textContent: "Chưa có dữ liệu đo cho chỉ số này.",
      }),
    );
    return;
  }

  const labelWidth = 180;
  const chartWidth = 430;
  const rowHeight = series.length > 1 ? 46 : 36;
  const height = 20 + measuredRows.length * rowHeight;
  const maxValue = options.maxValue ?? Math.max(
    ...measuredRows.flatMap((row) =>
      series.map((item) => row[item.key]).filter(Number.isFinite),
    ),
    0.0001,
  ) * 1.15;
  const svg = makeSvgElement("svg", {
    viewBox: `0 0 ${labelWidth + chartWidth + 80} ${height}`,
    role: "img",
    "aria-label": options.ariaLabel,
  });

  measuredRows.forEach((row, rowIndex) => {
    const y = 10 + rowIndex * rowHeight;
    const label = makeSvgElement("text", {
      x: 0,
      y: y + 13,
      fill: "#c5d0d1",
      "font-size": 10,
    });
    const fullName = row.model_name ?? row.name;
    label.textContent = fullName.length > 25
      ? `${fullName.slice(0, 23)}…`
      : fullName;
    const labelTitle = makeSvgElement("title");
    labelTitle.textContent = fullName;
    label.append(labelTitle);
    svg.append(label);

    series.forEach((item, seriesIndex) => {
      const value = row[item.key];
      if (!Number.isFinite(value)) return;
      const barY = y + seriesIndex * 17;
      const width = Math.max(2, (value / maxValue) * chartWidth);
      const rect = makeSvgElement("rect", {
        x: labelWidth,
        y: barY,
        width,
        height: 11,
        rx: 4,
        fill: item.color ?? colors[seriesIndex % colors.length],
        opacity: 0.9,
      });
      const title = makeSvgElement("title");
      title.textContent =
        `${fullName} · ${item.label}: ${item.format(value)}`;
      rect.append(title);
      svg.append(rect);

      const valueLabel = makeSvgElement("text", {
        x: labelWidth + width + 7,
        y: barY + 9,
        fill: "#eff5f5",
        "font-size": 9,
      });
      valueLabel.textContent = item.format(value);
      svg.append(valueLabel);
    });
  });
  container.append(svg);

  if (series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const [index, item] of series.entries()) {
      const entry = document.createElement("span");
      entry.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = item.color ?? colors[index % colors.length];
      const text = document.createElement("span");
      text.textContent = item.label;
      entry.append(swatch, text);
      legend.append(entry);
    }
    container.append(legend);
  }
}

function renderCharts(rows) {
  renderBarChart("chart-precision", rows, [
    { key: "precision", label: "Precision", format: formatPercent },
  ], { maxValue: 1, ariaLabel: "Biểu đồ Precision" });
  renderBarChart("chart-recall", rows, [
    { key: "recall", label: "Recall", format: formatPercent },
  ], { maxValue: 1, ariaLabel: "Biểu đồ Recall" });
  renderBarChart("chart-map", rows, [
    { key: "map50", label: "mAP50", format: formatPercent, color: colors[0] },
    { key: "map50_95", label: "mAP50–95", format: formatPercent, color: colors[1] },
  ], { maxValue: 1, ariaLabel: "Biểu đồ mAP50 và mAP50–95" });
  renderBarChart("chart-latency", rows, [
    {
      key: "inference_ms",
      label: "Thời gian",
      format: (value) => `${formatNumber(value, 2)} ms`,
      color: colors[2],
    },
  ], { ariaLabel: "Biểu đồ thời gian suy luận" });
  renderBarChart("chart-size", rows, [
    {
      key: "model_size_mb",
      label: "Kích thước",
      format: (value) => `${formatNumber(value, 2)} MB`,
      color: colors[1],
    },
  ], { ariaLabel: "Biểu đồ kích thước mô hình" });
}

function createStat(label, value, note) {
  const article = document.createElement("article");
  article.className = "benchmark-stat";
  const name = document.createElement("span");
  name.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const small = document.createElement("small");
  small.textContent = note;
  article.append(name, strong, small);
  return article;
}

function renderSummary(rows) {
  summary.replaceChildren();
  if (!rows.length) {
    summary.append(
      createStat("MÔ HÌNH", "—", "Chưa có mô hình được chọn"),
      createStat("TẬP ĐÁNH GIÁ", "—", "Chưa có dữ liệu"),
    );
    return;
  }

  const first = rows[0];
  const best = (key) => rows
    .filter((row) => Number.isFinite(row[key]))
    .sort((a, b) => b[key] - a[key])[0];
  const fastest = rows
    .filter((row) => Number.isFinite(row.inference_ms))
    .sort((a, b) => a.inference_ms - b.inference_ms)[0];
  const precisionModel = best("precision");
  const recallModel = best("recall");
  const mapModel = best("map50");
  summary.append(
    createStat(
      "PRECISION CAO NHẤT",
      precisionModel ? formatPercent(precisionModel.precision) : "Chưa có dữ liệu",
      precisionModel?.model_name ?? "Chưa đo",
    ),
    createStat(
      "RECALL CAO NHẤT",
      recallModel ? formatPercent(recallModel.recall) : "Chưa có dữ liệu",
      recallModel?.model_name ?? "Chưa đo",
    ),
    createStat(
      "mAP50 CAO NHẤT",
      mapModel ? formatPercent(mapModel.map50) : "Chưa có dữ liệu",
      mapModel?.model_name ?? "Chưa đo",
    ),
    createStat(
      "SUY LUẬN NHANH NHẤT",
      fastest ? `${formatNumber(fastest.inference_ms, 2)} ms` : "Chưa đo",
      fastest?.model_name ?? "Chưa có dữ liệu",
    ),
    createStat(
      "ẢNH ĐÁNH GIÁ",
      Number.isFinite(first.image_count) ? String(first.image_count) : "Chưa có dữ liệu",
      first.dataset ?? "Tập dữ liệu",
    ),
    createStat(
      "ĐỐI TƯỢNG CÓ NHÃN",
      Number.isFinite(first.object_count) ? String(first.object_count) : "Chưa có dữ liệu",
      "Trong tập được đánh giá",
    ),
    createStat(
      "KÍCH THƯỚC CHECKPOINT",
      Number.isFinite(first.model_size_mb)
        ? `${formatNumber(first.model_size_mb, 2)} MB`
        : "Chưa đo",
      first.model_name ?? "Mô hình",
    ),
    createStat(
      "THIẾT BỊ",
      first.device ?? "Chưa đo",
      first.timing_scope ?? "Thời gian suy luận",
    ),
  );
}

function renderResultsTable(rows) {
  resultsBody.replaceChildren();
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 11;
    cell.className = "not-evaluated";
    cell.textContent = "Chưa có kết quả đánh giá.";
    row.append(cell);
    resultsBody.append(row);
    return;
  }

  for (const result of rows) {
    const row = document.createElement("tr");
    const values = [
      result.model_name,
      result.task_label,
      formatPercent(result.precision),
      formatPercent(result.recall),
      formatPercent(result.map50),
      formatPercent(result.map50_95),
      Number.isFinite(result.inference_ms)
        ? formatNumber(result.inference_ms, 2)
        : "Chưa đo",
      Number.isFinite(result.model_size_mb)
        ? formatNumber(result.model_size_mb, 2)
        : "Chưa đo",
      Number.isFinite(result.image_count) && Number.isFinite(result.object_count)
        ? `${result.image_count} / ${result.object_count}`
        : "Chưa có dữ liệu",
      `${result.dataset ?? "Chưa có dữ liệu"} · ${result.split_label ?? result.split ?? "—"}`,
      result.status ?? "Chưa đánh giá",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = escapeText(value);
      row.append(cell);
    }
    resultsBody.append(row);
  }
}

function renderCommentary(rows, source) {
  commentary.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = "Nhận xét kết quả";
  commentary.append(title);
  if (!rows.length) {
    const text = document.createElement("p");
    text.textContent = "Chưa có số liệu để nhận xét.";
    commentary.append(text);
    return;
  }

  const notes = [];
  if (rows.length === 1) {
    notes.push(
      "Chỉ có một mô hình đã xác minh cho bài toán này; số liệu mô tả mô hình, chưa tạo thành so sánh giữa nhiều phiên bản.",
    );
  } else {
    const highest = (key) => rows
      .filter((row) => Number.isFinite(row[key]))
      .sort((a, b) => b[key] - a[key])[0];
    const fastest = rows
      .filter((row) => Number.isFinite(row.inference_ms))
      .sort((a, b) => a.inference_ms - b.inference_ms)[0];
    for (const [key, label] of [
      ["precision", "Precision"],
      ["recall", "Recall"],
      ["map50", "mAP50"],
    ]) {
      const model = highest(key);
      if (model) notes.push(`${label} cao nhất: ${model.model_name} (${formatPercent(model[key])}).`);
    }
    if (fastest) {
      notes.push(
        `Suy luận nhanh nhất: ${fastest.model_name} (${formatNumber(fastest.inference_ms, 2)} ms/ảnh).`,
      );
    }
  }
  const measuredSize = rows.filter((row) => Number.isFinite(row.model_size_mb));
  if (measuredSize.length) {
    notes.push(
      `Kích thước checkpoint: ${measuredSize.map((row) => `${row.model_name} ${formatNumber(row.model_size_mb, 2)} MB`).join("; ")}.`,
    );
  } else {
    notes.push("Chưa có dữ liệu kích thước checkpoint.");
  }
  if (rows.some((row) => !Number.isFinite(row.inference_ms))) {
    notes.push("Thời gian suy luận chưa được đo; các chỉ số này không được suy diễn từ metric độ chính xác.");
  }
  if (rows.some((row) => row.orphan_label_count > 0)) {
    notes.push("Tập dữ liệu có file nhãn không kèm ảnh; các file mồ côi được ghi nhận và không tính vào số ảnh/đối tượng đánh giá.");
  }
  if (source === "validation") {
    notes.push("Đây là số liệu validation trong lịch sử huấn luyện, không phải kết quả test độc lập.");
  } else {
    notes.push(
      "Kết quả phụ thuộc vào tập nhãn, confidence, thiết bị và kích thước ảnh; không dùng riêng các metric này làm căn cứ xử phạt.",
    );
  }

  const list = document.createElement("ul");
  for (const note of notes) {
    const item = document.createElement("li");
    item.textContent = note;
    list.append(item);
  }
  commentary.append(list);
}

function updateResults(rows, source, context) {
  resultsContext.textContent = context;
  renderSummary(rows);
  renderCharts(rows);
  renderResultsTable(rows);
  renderCommentary(rows, source);
}

function showTrainingBaseline() {
  const selectedIds = new Set(
    [...modelOptions.querySelectorAll('input[name="model-id"]:checked')]
      .map((input) => input.value),
  );
  const selected = selectedTaskModels().filter(
    (model) => selectedIds.has(model.id),
  );
  const rows = selected.map((model) => {
    const metrics = model.training_metrics;
    const splitData = model.splits?.val;
    return {
      model_name: model.name,
      task_label: model.task_label,
      precision: metrics?.precision,
      recall: metrics?.recall,
      map50: metrics?.map50,
      map50_95: metrics?.map50_95,
      inference_ms: null,
      model_size_mb: model.model_size_mb,
      image_count: splitData?.image_count,
      object_count: splitData?.object_count,
      dataset: model.dataset_label,
      split: "val",
      split_label: "validation · lịch sử huấn luyện",
      status: metrics
        ? `Validation huấn luyện · epoch ${metrics.epoch}`
        : "Chưa có metric validation",
    };
  });
  updateResults(rows, "validation", "Mốc validation đã ghi nhận trong huấn luyện");
}

async function loadHistory() {
  const query = new URLSearchParams({ task: taskSelect.value });
  try {
    const response = await fetch(`/api/ai-comparison/history?${query}`, {
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("Không tải được lịch sử đánh giá.");
    const payload = await response.json();
    historyBody.replaceChildren();
    if (!payload.results.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 7;
      cell.className = "not-evaluated";
      cell.textContent = "Chưa có lượt đánh giá được lưu.";
      row.append(cell);
      historyBody.append(row);
      return;
    }
    for (const result of payload.results) {
      const row = document.createElement("tr");
      const values = [
        new Date(result.evaluated_at).toLocaleString("vi-VN"),
        result.model_name,
        `${result.dataset} · ${result.split}`,
        formatPercent(result.map50),
        formatPercent(result.map50_95),
        `${formatNumber(result.inference_ms, 2)} ms`,
        result.device,
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = escapeText(value);
        row.append(cell);
      }
      historyBody.append(row);
    }
  } catch (error) {
    historyBody.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "not-evaluated";
    cell.textContent = error.message;
    row.append(cell);
    historyBody.append(row);
  }
}

async function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForJob(jobId) {
  const started = Date.now();
  while (Date.now() - started < 30 * 60 * 1000) {
    await sleep(1200);
    const response = await fetch(`/api/ai-comparison/jobs/${jobId}`, {
      credentials: "same-origin",
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error ?? "Không đọc được trạng thái đánh giá.");
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(job.error);
  }
  throw new Error("Đánh giá vượt quá thời gian chờ. Hãy kiểm tra log máy chủ.");
}

async function runComparison() {
  errorBox.hidden = true;
  const modelIds = [
    ...modelOptions.querySelectorAll('input[name="model-id"]:checked'),
  ].map((input) => input.value);
  if (!modelIds.length) {
    errorBox.textContent = "Không có checkpoint khả dụng được chọn.";
    errorBox.hidden = false;
    return;
  }

  const payload = {
    task: taskSelect.value,
    model_ids: modelIds,
    split: splitSelect.value,
    confidence: Number(confidenceInput.value),
    repeats: Number(repeatsInput.value),
  };
  if (!Number.isInteger(payload.repeats) || payload.repeats < 1 || payload.repeats > 5) {
    errorBox.textContent = "Số lượt đo phải là số nguyên từ 1 đến 5.";
    errorBox.hidden = false;
    repeatsInput.focus();
    return;
  }

  setConfigurationDisabled(true);
  runButton.disabled = true;
  loading.hidden = false;
  runButton.querySelector("span").textContent = "Đang chạy…";
  try {
    const response = await fetch("/api/ai-comparison/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error ?? "Không thể bắt đầu đánh giá.");
    const completed = await waitForJob(job.id);
    const results = completed.results.map((result) => ({
      ...result,
      split_label: result.split === "test" ? "test độc lập" : "validation",
    }));
    updateResults(
      results,
      "evaluation",
      `${results[0].dataset} · ${results[0].split_label} · ${results[0].device}`,
    );
    await loadHistory();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    setConfigurationDisabled(false);
    runButton.disabled = !selectedTaskModels().some((model) => model.available);
    loading.hidden = true;
    runButton.querySelector("span").textContent = "Chạy so sánh";
  }
}

function handleTaskChange() {
  renderModelOptions();
  showTrainingBaseline();
  loadHistory();
}

function setConfigurationDisabled(disabled) {
  taskSelect.disabled = disabled;
  splitSelect.disabled = disabled;
  confidenceInput.disabled = disabled;
  repeatsInput.disabled = disabled;
  modelOptions.querySelectorAll("input").forEach((input) => {
    input.disabled = disabled || !selectedTaskModels().some(
      (model) => model.id === input.value && model.available,
    );
  });
}

taskSelect.addEventListener("change", handleTaskChange);
splitSelect.addEventListener("change", updateDatasetInfo);
confidenceInput.addEventListener("input", () => {
  confidenceValue.value = `${Math.round(Number(confidenceInput.value) * 100)}%`;
});
modelOptions.addEventListener("change", showTrainingBaseline);
runButton.addEventListener("click", runComparison);

renderModelOptions();
showTrainingBaseline();
loadHistory();
