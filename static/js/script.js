const uploadSection = document.getElementById('upload-section');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const removeImageBtn = document.getElementById('remove-image');
const predictBtn = document.getElementById('predict-btn');
const resultsSection = document.getElementById('results-section');
const resultContent = document.getElementById('result-content');
const loadingSpinner = document.getElementById('loading-spinner');

// Drag and drop handling
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('border-blue-500');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('border-blue-500');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('border-blue-500');
    handleFile(e.dataTransfer.files[0]);
});

// File input handling
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Remove image handling
removeImageBtn.addEventListener('click', () => {
    resetUI();
});

// Predict button handling
predictBtn.addEventListener('click', async () => {
    try {
        showLoading(true);
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        showError('An error occurred during analysis. Please try again.');
    } finally {
        showLoading(false);
    }
});

// Helper functions
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Please upload a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
        predictBtn.disabled = false;
        resultsSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUI() {
    fileInput.value = '';
    previewImage.src = '';
    previewSection.classList.add('hidden');
    predictBtn.disabled = true;
    resultsSection.classList.add('hidden');
}

function showLoading(show) {
    loadingSpinner.classList.toggle('hidden', !show);
    predictBtn.disabled = show;
}

function displayResults(result) {
    resultsSection.classList.remove('hidden');
    
    const tumorTypes = {
        glioma: 'Glioma',
        meningioma: 'Meningioma',
        notumor: 'No Tumor',
        pituitary: 'Pituitary Tumor'
    };

    const resultHTML = `
        <div class="${result.prediction === 'notumor' ? 'bg-green-100' : 'bg-yellow-100'} rounded-lg p-4">
            <p class="text-lg font-medium ${result.prediction === 'notumor' ? 'text-green-800' : 'text-yellow-800'}">
                ${tumorTypes[result.prediction]}
            </p>
            <p class="text-sm mt-2 ${result.prediction === 'notumor' ? 'text-green-600' : 'text-yellow-600'}">
                Confidence: ${(result.probability * 100).toFixed(2)}%
            </p>
        </div>
    `;

    resultContent.innerHTML = resultHTML;
}

function showError(message) {
    resultsSection.classList.remove('hidden');
    resultContent.innerHTML = `
        <div class="bg-red-100 text-red-700 p-4 rounded-lg">
            <p>${message}</p>
        </div>
    `;
}

// Add file type validation
const allowedExtensions = ['jpg', 'jpeg', 'png'];

function handleFileUpload(file) {
    const extension = file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(extension)) {
        showError('Invalid file type. Please upload JPG/JPEG/PNG');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
        } else {
            showResult(data);
        }
    })
    .catch(error => {
        showError('Upload failed: ' + error.message);
    });
}