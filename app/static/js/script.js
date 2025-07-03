
const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    webcam.srcObject = stream;
  })
  .catch((err) => {
    console.error("Error accessing webcam: ", err);
    alert("Webcam access is blocked.");
  });

async function captureFromCamera() {
  const context = canvas.getContext("2d");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "captured.png");
    const res = await fetch("/upload-image/", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    document.getElementById(
      "captureResult"
    ).innerText = `Captured Person is:  ${data.result.person}`;
  }, "image/png");
}

function showSection(id) {
  document
    .querySelectorAll(".section")
    .forEach((el) => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

const uploadBox = document.getElementById("uploadBox");
uploadBox.addEventListener("click", () =>
  document.getElementById("uploadFile").click()
);
uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.style.borderColor = "#6200ea";
});
uploadBox.addEventListener("dragleave", () => {
  uploadBox.style.borderColor = "#2575fc";
});
uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  document.getElementById("uploadFile").files = e.dataTransfer.files;
});

async function uploadImage() {
  const fileInput = document.getElementById("uploadFile");
  const resultBox = document.getElementById("uploadResult");

  // Clear previous result
  resultBox.innerText = "";

  if (!fileInput.files.length) {
    alert("Please select an image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/upload-image/", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errorText = await res.text(); // fallback if not JSON
      throw new Error(`Server error ${res.status}: ${errorText}`);
    }

    const data = await res.json();

    if (data.result && data.result.person) {
      resultBox.innerText = `The image person is: ${data.result.person}`;
    } else if (data.result) {
      resultBox.innerText = `Result: ${JSON.stringify(data.result)}`;
    } else if (data.error) {
      resultBox.innerText = `Error: ${data.error}`;
    } else {
      resultBox.innerText = "Unexpected response format.";
    }
  } catch (err) {
    console.error("Upload error:", err);
    resultBox.innerText = `❌ Upload failed: ${err.message}`;
  }
}

// async function uploadDatasetAndTrain() {
//   const folderInput = document.getElementById("datasetFolder");
//   const labelName = document.getElementById("labelName").value.trim();
//   if (!labelName || !folderInput.files.length)
//     return alert("Provide label and images.");

//   let progress = document.getElementById("progressFill");
//   let uploaded = 0;

//   for (const file of folderInput.files) {
//     const formData = new FormData();
//     formData.append("file", file);
//     formData.append("label", labelName);
//     await fetch(`/upload-dataset/${labelName}/`, {
//       method: "POST",
//       body: formData,
//     });
//     uploaded++;
//     progress.style.width =
//       ((uploaded / folderInput.files.length) * 100).toFixed(0) + "%";
//     progress.innerText = progress.style.width;
//   }

//   const trainRes = await fetch("/train/", { method: "POST" });
//   const result = await trainRes.json();
//   document.getElementById(
//     "trainResult"
//   ).innerText = `Training Completed: ${result.faces} face(s)`;
// }

async function uploadDatasetAndTrain() {
  const folderInput = document.getElementById("datasetFolder");
  const labelName = document.getElementById("labelName").value.trim();
  const logBox = document.getElementById("trainResult");

  if (!labelName || !folderInput.files.length) {
    alert("Provide label and images.");
    return;
  }

  logBox.innerText = "Uploading files...\n";
  let progress = document.getElementById("progressFill");
  let uploaded = 0;

  for (const file of folderInput.files) {
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`/upload-dataset/${labelName}/`, {
      method: "POST",
      body: formData,
    });
    uploaded++;
    const percent =
      ((uploaded / folderInput.files.length) * 100).toFixed(0) + "%";
    progress.style.width = percent;
    progress.innerText = percent;
  }

  logBox.innerText += "✅ Upload complete. Starting training...\n";

  const response = await fetch("/train/", { method: "POST" });
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    logBox.innerText += decoder.decode(value);
    logBox.scrollTop = logBox.scrollHeight; // Auto-scroll
  }

  logBox.innerText += "\n✅ Training Completed!";
}
