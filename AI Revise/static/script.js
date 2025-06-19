document.addEventListener("DOMContentLoaded", function() {
  const fileInput = document.querySelector('input[type="file"]');
  const predictButton = document.getElementById('predict-btn');

  // Enable the predict button when a file is selected
  fileInput.addEventListener('change', function() {
      if (fileInput.files.length > 0) {
          predictButton.disabled = false;
      } else {
          predictButton.disabled = true;
      }
  });
});
