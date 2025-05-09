document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const dropArea = document.getElementById('drop-area');
    const imagePreview = document.getElementById('image-preview');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const defectClass = document.getElementById('defect-class');
    const predictionDetails = document.getElementById('prediction-details');
    const defectInfo = document.getElementById('defect-info');
    const allModelsCheckbox = document.getElementById('all-models');
    const modelCheckboxes = document.querySelectorAll('input[name="models"]:not(#all-models)');

    // Handle drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('drag-over');
    }

    function unhighlight() {
        dropArea.classList.remove('drag-over');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            imageUpload.files = files;
            updateImagePreview(files[0]);
        }
    }

    // Update image preview when a file is selected
    imageUpload.addEventListener('change', function() {
        if (this.files.length) {
            updateImagePreview(this.files[0]);
        }
    });

    function updateImagePreview(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Handle "All Models" checkbox
    allModelsCheckbox.addEventListener('change', function() {
        if (this.checked) {
            modelCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
                checkbox.disabled = true;
            });
        } else {
            modelCheckboxes.forEach(checkbox => {
                checkbox.disabled = false;
                checkbox.checked = true;
            });
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Check if an image is selected
        if (!imageUpload.files.length) {
            alert('Please select an image first');
            return;
        }
        
        // Check if at least one model is selected
        const selectedModels = document.querySelectorAll('input[name="models"]:checked');
        if (selectedModels.length === 0) {
            alert('Please select at least one model');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('file', imageUpload.files[0]);
        
        // Add selected models to formData
        selectedModels.forEach(model => {
            formData.append('models', model.value);
        });
        
        // Show loading indicator
        document.getElementById('detect-btn').innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        document.getElementById('detect-btn').disabled = true;
        
        // Send request to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Display the results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
        })
        .finally(() => {
            // Reset button
            document.getElementById('detect-btn').innerHTML = '<i class="fas fa-search me-2"></i>Detect Defect';
            document.getElementById('detect-btn').disabled = false;
        });
    });

    function displayResults(data) {
        // Set the result image
        resultImage.src = data.filepath;
        
        // Set the detected defect class
        defectClass.textContent = data.defect_class;
        
        // Clear previous prediction details
        predictionDetails.innerHTML = '';
        
        // Display model predictions
        const models = Object.keys(data.model_predictions);
        
        models.forEach(model => {
            const modelData = data.model_predictions[model];
            const modelDiv = document.createElement('div');
            modelDiv.className = 'model-prediction mb-3';
            
            // Create model header
            const modelHeader = document.createElement('h6');
            modelHeader.textContent = formatModelName(model);
            modelDiv.appendChild(modelHeader);
            
            // Create prediction visualization
            const predictionContent = document.createElement('div');
            predictionContent.className = 'prediction-content';
            
            // Get top 3 predictions for this model
            const predictions = Object.entries(modelData.probabilities)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3);
            
            predictions.forEach(([className, probability]) => {
                const percentage = (probability * 100).toFixed(1);
                
                const predBar = document.createElement('div');
                predBar.className = 'mb-2';
                
                const predLabel = document.createElement('div');
                predLabel.className = 'prediction-label';
                predLabel.innerHTML = `
                    <span>${className}</span>
                    <span>${percentage}%</span>
                `;
                
                const progress = document.createElement('div');
                progress.className = 'progress';
                progress.style.height = '10px';
                
                const progressBar = document.createElement('div');
                progressBar.className = `progress-bar ${className === modelData.class ? 'bg-success' : 'bg-primary'}`;
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute('role', 'progressbar');
                progressBar.setAttribute('aria-valuenow', percentage);
                progressBar.setAttribute('aria-valuemin', '0');
                progressBar.setAttribute('aria-valuemax', '100');
                
                progress.appendChild(progressBar);
                predBar.appendChild(predLabel);
                predBar.appendChild(progress);
                predictionContent.appendChild(predBar);
            });
            
            modelDiv.appendChild(predictionContent);
            predictionDetails.appendChild(modelDiv);
        });
        
        // Display defect information
        defectInfo.innerHTML = `
            <div class="defect-info">
                <h3>Defect Information</h3>
                <div class="info-section">
                    <h4>Description</h4>
                    <p>${data.defect_info.description}</p>
                </div>
                <div class="info-section">
                    <h4>Common Causes</h4>
                    <p>${data.defect_info.causes}</p>
                </div>
                <div class="info-section">
                    <h4>Prevention Methods</h4>
                    <p>${data.defect_info.prevention}</p>
                </div>
            </div>
        `;
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function formatModelName(modelName) {
        switch(modelName) {
            case 'resnet50':
                return 'ResNet50';
            case 'efficientnet':
                return 'EfficientNetB0';
            case 'mobilenet':
                return 'MobileNetV2';
            case 'custom':
                return 'Custom CNN';
            default:
                return modelName.charAt(0).toUpperCase() + modelName.slice(1);
        }
    }
}); 