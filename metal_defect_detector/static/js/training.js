document.addEventListener('DOMContentLoaded', function() {
    // Elements for Upload Data tab
    const uploadTrainingForm = document.getElementById('upload-training-form');
    const uploadStatus = document.getElementById('upload-status');
    
    // Elements for Preprocess Data tab
    const preprocessForm = document.getElementById('preprocess-form');
    const preprocessStatus = document.getElementById('preprocess-status');
    
    // Elements for Train Model tab
    const trainModelForm = document.getElementById('train-model-form');
    const trainingStatus = document.getElementById('training-status');
    const trainingProgress = document.getElementById('training-progress');
    const progressBar = document.querySelector('.progress-bar');
    const currentEpoch = document.getElementById('current-epoch');
    const totalEpochs = document.getElementById('total-epochs');
    const currentLoss = document.getElementById('current-loss');
    const currentAccuracy = document.getElementById('current-accuracy');
    
    // Initialize charts
    initCharts();
    
    // Handle Upload Training Form submission
    if (uploadTrainingForm) {
        uploadTrainingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Check if category is selected
            const category = document.getElementById('defect-category').value;
            if (!category) {
                alert('Please select a defect category');
                return;
            }
            
            // Check if files are selected
            const trainingImages = document.getElementById('training-images').files;
            if (trainingImages.length === 0) {
                alert('Please select at least one image to upload');
                return;
            }
            
            // Create FormData object
            const formData = new FormData();
            formData.append('category', category);
            
            // Add all selected files to formData
            for (let i = 0; i < trainingImages.length; i++) {
                formData.append('images', trainingImages[i]);
            }
            
            // Show uploading status
            uploadStatus.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    Uploading ${trainingImages.length} images to the '${category}' category...
                </div>
            `;
            uploadStatus.style.display = 'block';
            
            // In a real implementation, we would send this data to the server
            // For this demo, we'll simulate a successful upload after a delay
            setTimeout(() => {
                uploadStatus.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Successfully uploaded ${trainingImages.length} images to the '${category}' category!
                    </div>
                `;
                
                // Reset the form
                uploadTrainingForm.reset();
                
                // Update the data distribution chart
                updateDataDistributionChart(category, trainingImages.length);
            }, 2000);
        });
    }
    
    // Handle Preprocess Form submission
    if (preprocessForm) {
        preprocessForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get selected transformations
            const transformations = Array.from(document.querySelectorAll('input[name="transformations"]:checked'))
                .map(checkbox => checkbox.value);
            
            // Get selected preprocessing options
            const preprocessing = Array.from(document.querySelectorAll('input[name="preprocessing"]:checked'))
                .map(checkbox => checkbox.value);
            
            // Get augmentation factor
            const augmentationFactor = document.getElementById('augmentation-factor').value;
            
            // Show preprocessing status
            preprocessStatus.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    Preprocessing dataset with selected options...
                </div>
            `;
            preprocessStatus.style.display = 'block';
            
            // In a real implementation, we would send this data to the server
            // For this demo, we'll simulate processing after a delay
            setTimeout(() => {
                const originalCount = 251; // Our metal defect dataset size
                const newCount = originalCount * parseInt(augmentationFactor);
                
                preprocessStatus.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Successfully preprocessed the dataset! Original size: ${originalCount} images, New size: ${newCount} images.
                        <br>
                        Applied transformations: ${transformations.join(', ')}
                        <br>
                        Applied preprocessing: ${preprocessing.join(', ')}
                    </div>
                `;
            }, 3000);
        });
    }
    
    // Handle Train Model Form submission
    if (trainModelForm) {
        trainModelForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get training parameters
            const modelType = document.getElementById('model-type').value;
            const batchSize = document.getElementById('batch-size').value;
            const epochs = document.getElementById('epochs').value;
            const learningRate = document.getElementById('learning-rate').value;
            const useTransferLearning = document.getElementById('use-transfer-learning').checked;
            
            // Show training status
            trainingStatus.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    Initializing ${modelType} model training with batch size ${batchSize}, learning rate ${learningRate}, and ${epochs} epochs...
                </div>
            `;
            trainingStatus.style.display = 'block';
            
            // Show training progress
            trainingProgress.style.display = 'block';
            totalEpochs.textContent = epochs;
            
            // In a real implementation, we would start the training process on the server
            // For this demo, we'll simulate training progress updates
            simulateTraining(parseInt(epochs));
        });
    }
    
    function simulateTraining(epochs) {
        let currentEpochNum = 0;
        const trainingInterval = setInterval(() => {
            currentEpochNum++;
            
            // Update progress
            const progress = (currentEpochNum / epochs) * 100;
            progressBar.style.width = `${progress}%`;
            currentEpoch.textContent = currentEpochNum;
            
            // Generate random loss and accuracy values (decreasing loss, increasing accuracy)
            const loss = Math.max(0.05, 0.5 - (0.45 * currentEpochNum / epochs) + (Math.random() * 0.1 - 0.05)).toFixed(4);
            const accuracy = Math.min(0.99, 0.7 + (0.29 * currentEpochNum / epochs) + (Math.random() * 0.05 - 0.025)).toFixed(4);
            
            currentLoss.textContent = loss;
            currentAccuracy.textContent = `${(parseFloat(accuracy) * 100).toFixed(2)}%`;
            
            // Update training performance chart
            updatePerformanceChart(currentEpochNum, parseFloat(loss), parseFloat(accuracy));
            
            // If training is complete
            if (currentEpochNum >= epochs) {
                clearInterval(trainingInterval);
                
                // Show completion message
                trainingStatus.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Model training completed successfully!
                        <br>
                        Final loss: ${loss}, Final accuracy: ${(parseFloat(accuracy) * 100).toFixed(2)}%
                        <br>
                        The model has been saved and is ready for use.
                    </div>
                `;
            }
        }, 500); // Update every 500ms for simulation
    }
    
    function initCharts() {
        // Data Distribution Chart
        const dataDistributionCtx = document.getElementById('data-distribution-chart');
        if (dataDistributionCtx) {
            window.dataDistributionChart = new Chart(dataDistributionCtx, {
                type: 'bar',
                data: {
                    labels: ['Crazing', 'Scratches', 'inclusion', 'oil spot', 'water spot', 
                        'welding line', 'waist folding', 'silk spot', 'punching', 
                        'crease', 'crescent gap', 'rolling'],
                    datasets: [{
                        label: 'Number of Images',
                        data: [30, 30, 22, 21, 28, 15, 28, 22, 22, 6, 23, 4],
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Dataset Distribution by Defect Category'
                        }
                    }
                }
            });
        }
        
        // Training Performance Chart
        const trainingPerformanceCtx = document.getElementById('training-performance-chart');
        if (trainingPerformanceCtx) {
            window.trainingPerformanceChart = new Chart(trainingPerformanceCtx, {
                type: 'line',
                data: {
                    labels: [],  // Will be filled with epoch numbers
                    datasets: [
                        {
                            label: 'Loss',
                            data: [],  // Will be filled with loss values
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Accuracy',
                            data: [],  // Will be filled with accuracy values
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Loss'
                            },
                            min: 0
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Accuracy'
                            },
                            min: 0,
                            max: 1,
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Training Performance'
                        }
                    }
                }
            });
        }
    }
    
    function updateDataDistributionChart(category, count) {
        // Map category to index in the chart
        const categoryIndex = {
            'Crazing': 0,
            'Scratches': 1,
            'inclusion': 2,
            'oil spot': 3,
            'water spot': 4,
            'welding line': 5,
            'waist folding': 6,
            'silk spot': 7,
            'punching': 8,
            'crease': 9,
            'crescent gap': 10,
            'rolling': 11
        }[category];
        
        // Update the dataset
        if (window.dataDistributionChart && categoryIndex !== undefined) {
            let currentCount = window.dataDistributionChart.data.datasets[0].data[categoryIndex];
            window.dataDistributionChart.data.datasets[0].data[categoryIndex] = currentCount + count;
            window.dataDistributionChart.update();
        }
    }
    
    function updatePerformanceChart(epoch, loss, accuracy) {
        if (window.trainingPerformanceChart) {
            // Add the epoch label if it doesn't exist
            if (window.trainingPerformanceChart.data.labels.length < epoch) {
                window.trainingPerformanceChart.data.labels.push(epoch);
            }
            
            // Add the loss and accuracy data
            window.trainingPerformanceChart.data.datasets[0].data.push(loss);
            window.trainingPerformanceChart.data.datasets[1].data.push(accuracy);
            
            // Update the chart
            window.trainingPerformanceChart.update();
        }
    }
}); 