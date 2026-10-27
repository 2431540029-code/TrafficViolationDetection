const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const form = document.querySelector("#upload-form");
const submitButton = document.querySelector("#submit-button");
const results = document.querySelector("#results");
const historySection = document.querySelector("#history");

if (results) {
  requestAnimationFrame(() => {
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function showFile(file) {
  if (!file) return;
  document.querySelector("#file-title").textContent = file.name;
  document.querySelector("#file-description").textContent =
    `${(file.size / (1024 * 1024)).toFixed(2)} MB · Hãy chọn ảnh raster hợp lệ`;
}

input.addEventListener("change", () => showFile(input.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
}

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  showFile(file);
});

form.addEventListener("submit", () => {
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Đang nhận diện...";
});

if (historySection) {
  const historyError = historySection.querySelector(".history-delete-error");

  historySection.addEventListener("submit", async (event) => {
    const deleteForm = event.target.closest("[data-history-delete]");
    if (!deleteForm) return;

    event.preventDefault();
    const button = deleteForm.querySelector("button");
    button.disabled = true;
    historyError.hidden = true;

    try {
      const response = await fetch(deleteForm.action, {
        method: "POST",
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error(`Không xóa được lịch sử (HTTP ${response.status}).`);
      }

      const result = await response.json();
      if (!result.deleted) {
        throw new Error("Máy chủ chưa xác nhận đã xóa lịch sử.");
      }

      deleteForm.closest(".history-row").remove();
      const historyList = historySection.querySelector(".history-list");
      if (historyList && !historyList.querySelector(".history-row")) {
        historyList.replaceWith(
          Object.assign(document.createElement("p"), {
            className: "history-empty",
            textContent:
              "Chưa có ảnh nào được phân tích. Kết quả mới sẽ tự động lưu tại đây.",
          }),
        );
      }
    } catch (error) {
      historyError.textContent = error.message;
      historyError.hidden = false;
      button.disabled = false;
    }
  });
}
