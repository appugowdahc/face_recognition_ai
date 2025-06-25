const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const preview = document.getElementById('preview');
const fileInput = document.getElementById('fileInput');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => console.error("Webcam error:", err));

// Capture from webcam and upload
snap.addEventListener('click', () => {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    const preview = document.getElementById('preview');
    const imageUrl = URL.createObjectURL(blob);
    preview.src = imageUrl; // ✅ set image blob as src
    preview.style.display = 'block';

    // Upload image
    const formData = new FormData();
    formData.append("file", blob, "captured.png");
    sendToAPI(formData);
  }, 'image/png');
});

// Upload selected file
function uploadFile() {
  const file = fileInput.files[0];
  if (!file) return alert("Please select a file.");
  preview.src = URL.createObjectURL(file);
  const formData = new FormData();
  formData.append("file", file);
  sendToAPI(formData);
}

// Upload captured blob
function uploadBlob(blob, name) {
  const formData = new FormData();
  formData.append("file", blob, name);
  sendToAPI(formData);
}

// Send to FastAPI backend
function sendToAPI(formData) {
  fetch('/api/upload/', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => alert("✅ " + data.message + "\n🆔: " + data.id))
  .catch(err => alert("Upload failed"));
}
