// Global state
let currentView = 'comparison';
let currentModel = null;
let uploadedImage = null;
let availableModels = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadModels();
});

// Event Listeners
function initializeEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', handleNavigation);
    });

    // Image upload - Fix file input button
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');
    const uploadBtn = document.getElementById('upload-btn');
    const clearImageBtn = document.getElementById('clear-image-btn');

    // Multiple ways to trigger file input
    if (uploadArea) {
        uploadArea.addEventListener('click', (e) => {
            // Don't trigger if clicking the button inside
            if (!e.target.closest('.btn-primary')) {
                imageInput.click();
            }
        });
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.click();
        });
    }

    if (imageInput) {
        imageInput.addEventListener('change', handleImageSelect);
    }

    if (clearImageBtn) {
        clearImageBtn.addEventListener('click', clearImage);
    }

    // Drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
}

// Navigation
function handleNavigation(e) {
    const item = e.currentTarget;
    
    // Don't navigate if item is disabled/unavailable
    if (item.disabled || item.classList.contains('unavailable')) {
        return;
    }
    
    const view = item.dataset.view;
    const model = item.dataset.model;

    // Update active state
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
    item.classList.add('active');

    if (view === 'comparison') {
        currentView = 'comparison';
        currentModel = null;
        showComparisonView();
    } else if (view === 'about') {
        currentView = 'about';
        currentModel = null;
        showAboutView();
    } else if (model) {
        currentView = 'model';
        currentModel = model;
        showModelView(model);
    }

    // If image is uploaded, make prediction
    if (uploadedImage) {
        if (view === 'comparison') {
            predictAll();
        } else if (model) {
            predictSingle(model);
        }
    }
}

// Image Handling
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.classList.add('dragover');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleImageFile(files[0]);
    }
}

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

function handleImageFile(file) {
    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        alert('File size exceeds 10MB. Please choose a smaller image.');
        return;
    }

    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage = file;
        showImagePreview(e.target.result, file);
        
        // Auto-predict based on current view
        if (currentView === 'comparison') {
            predictAll();
        } else if (currentModel) {
            predictSingle(currentModel);
        }
    };
    reader.onerror = () => {
        alert('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
}

function showImagePreview(src, file) {
    const uploadArea = document.getElementById('upload-area');
    const preview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const imageName = document.getElementById('image-name');
    const imageSize = document.getElementById('image-size');

    if (uploadArea) uploadArea.style.display = 'none';
    if (preview) {
        preview.style.display = 'block';
        if (previewImg) previewImg.src = src;
        if (imageName) imageName.textContent = `File: ${file.name}`;
        if (imageSize) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            imageSize.textContent = `Size: ${sizeMB} MB`;
        }
    }
}

function clearImage() {
    uploadedImage = null;
    const uploadArea = document.getElementById('upload-area');
    const preview = document.getElementById('image-preview');
    const imageInput = document.getElementById('image-input');

    if (uploadArea) uploadArea.style.display = 'block';
    if (preview) preview.style.display = 'none';
    if (imageInput) imageInput.value = '';
    
    // Clear results
    const resultsGrid = document.getElementById('results-grid');
    const predictionResults = document.getElementById('prediction-results');
    
    if (resultsGrid) {
        resultsGrid.innerHTML = `
            <div class="empty-state-card">
                <i class="fas fa-upload"></i>
                <h4>No Image Uploaded</h4>
                <p>Upload an image above to see predictions from all models</p>
            </div>
        `;
    }
    
    if (predictionResults) {
        predictionResults.innerHTML = `
            <div class="empty-state-card">
                <i class="fas fa-upload"></i>
                <h4>No Image Uploaded</h4>
                <p>Upload an image to see predictions</p>
            </div>
        `;
    }
}

// Views
function showComparisonView() {
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const comparisonView = document.getElementById('comparison-view');
    const modelView = document.getElementById('model-view');
    const aboutView = document.getElementById('about-view');

    if (pageTitle) pageTitle.textContent = 'Model Comparison Dashboard';
    if (pageSubtitle) pageSubtitle.textContent = 'Upload an image to analyze predictions across multiple deep learning models';
    if (comparisonView) comparisonView.style.display = 'block';
    if (modelView) modelView.style.display = 'none';
    if (aboutView) aboutView.style.display = 'none';
}

function showModelView(modelId) {
    const modelName = getModelName(modelId);
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const modelNameDisplay = document.getElementById('model-name-display');
    const comparisonView = document.getElementById('comparison-view');
    const modelView = document.getElementById('model-view');
    const aboutView = document.getElementById('about-view');

    if (pageTitle) pageTitle.textContent = `${modelName} Analysis`;
    if (pageSubtitle) pageSubtitle.textContent = 'Detailed predictions and performance metrics';
    if (modelNameDisplay) modelNameDisplay.innerHTML = `<i class="fas fa-microscope"></i> ${modelName}`;
    if (comparisonView) comparisonView.style.display = 'none';
    if (modelView) modelView.style.display = 'block';
    if (aboutView) aboutView.style.display = 'none';
    
    loadModelMetrics(modelId);
}

function showAboutView() {
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const comparisonView = document.getElementById('comparison-view');
    const modelView = document.getElementById('model-view');
    const aboutView = document.getElementById('about-view');

    if (pageTitle) pageTitle.textContent = 'About This Project';
    if (pageSubtitle) pageSubtitle.textContent = 'Project details, models, dataset, and technology stack';
    if (comparisonView) comparisonView.style.display = 'none';
    if (modelView) modelView.style.display = 'none';
    if (aboutView) aboutView.style.display = 'block';
}

// API Calls
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error('Failed to load models');
        }
        const models = await response.json();
        availableModels = models;
        console.log('Available models:', models);
        updateModelAvailability(models);
        updateStats(models);
    } catch (error) {
        console.error('Error loading models:', error);
        // Show error but don't break the app
    }
}

function updateModelAvailability(models) {
    models.forEach(model => {
        const navItem = document.querySelector(`[data-model="${model.id}"]`);
        if (navItem) {
            const statusIcon = navItem.querySelector('.nav-status');
            if (!model.available) {
                navItem.classList.add('unavailable');
                navItem.title = 'Model file not found';
                navItem.disabled = true;
                if (statusIcon) {
                    statusIcon.className = 'nav-status unavailable-icon';
                    statusIcon.className = 'fas fa-times-circle nav-status unavailable-icon';
                }
            } else {
                navItem.classList.remove('unavailable');
                navItem.title = '';
                navItem.disabled = false;
                if (statusIcon) {
                    statusIcon.className = 'fas fa-check-circle nav-status available-icon';
                }
            }
        }
    });
    
    // Update available models count
    const availableCount = models.filter(m => m.available).length;
    const badge = document.getElementById('available-models-count');
    if (badge) {
        badge.textContent = availableCount;
    }
}

function updateStats(models) {
    const availableCount = models.filter(m => m.available).length;
    const statModels = document.getElementById('stat-models');
    if (statModels) {
        statModels.textContent = availableCount;
    }
}

async function loadModelMetrics(modelId) {
    try {
        const response = await fetch(`/api/models/${modelId}/metrics`);
        if (!response.ok) {
            throw new Error('Failed to load metrics');
        }
        const metrics = await response.json();
        displayMetrics(metrics);
    } catch (error) {
        console.error('Error loading metrics:', error);
        const container = document.getElementById('model-metrics');
        if (container) {
            container.innerHTML = '<p style="color: var(--error);">Failed to load metrics</p>';
        }
    }
}

function displayMetrics(metrics) {
    const container = document.getElementById('model-metrics');
    if (!container) return;

    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">${(metrics.f1_score * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Inference Time</div>
            <div class="metric-value">${metrics.inference_time}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Parameters</div>
            <div class="metric-value">${metrics.parameters}</div>
        </div>
    `;
}

async function predictAll() {
    if (!uploadedImage) return;

    const loading = document.getElementById('loading');
    const resultsGrid = document.getElementById('results-grid');
    
    if (loading) loading.style.display = 'flex';
    if (resultsGrid) resultsGrid.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('image', uploadedImage);

        const response = await fetch('/api/predict/all', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        if (data.results && data.results.length > 0) {
            displayComparisonResults(data.results);
        } else {
            if (resultsGrid) {
                resultsGrid.innerHTML = `
                    <div class="empty-state-card">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h4>No Models Available</h4>
                        <p>Please add model files to the models/ directory</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Prediction error:', error);
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="empty-state-card">
                    <i class="fas fa-exclamation-circle"></i>
                    <h4>Error</h4>
                    <p style="color: var(--error);">${error.message}</p>
                </div>
            `;
        }
    } finally {
        if (loading) loading.style.display = 'none';
    }
}

async function predictSingle(modelId) {
    if (!uploadedImage) return;

    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('prediction-results');
    
    if (loading) loading.style.display = 'flex';
    if (resultsContainer) resultsContainer.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('image', uploadedImage);
        formData.append('model_id', modelId);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        displaySingleResult(data);
    } catch (error) {
        console.error('Prediction error:', error);
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="empty-state-card">
                    <i class="fas fa-exclamation-circle"></i>
                    <h4>Error</h4>
                    <p style="color: var(--error);">${error.message}</p>
                </div>
            `;
        }
    } finally {
        if (loading) loading.style.display = 'none';
    }
}

// Display Results
function displayComparisonResults(results) {
    const grid = document.getElementById('results-grid');
    if (!grid) return;
    
    if (results.length === 0) {
        grid.innerHTML = `
            <div class="empty-state-card">
                <i class="fas fa-exclamation-triangle"></i>
                <h4>No Results</h4>
                <p>No models returned predictions</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = results.map(result => createModelCard(result)).join('');
    
    // Animate confidence bars
    setTimeout(() => {
        const bars = document.querySelectorAll('.confidence-fill');
        bars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 100);
}

function createModelCard(result) {
    const topPred = result.top_prediction;
    const maxConfidence = Math.max(...result.predictions.map(p => p.confidence));
    
    return `
        <div class="model-card">
            <div class="model-card-header">
                <div class="model-card-title">${result.model_name}</div>
                <div class="inference-time">
                    <i class="fas fa-clock"></i> ${result.inference_time}
                </div>
            </div>
            ${result.predictions.map((pred, idx) => `
                <div class="prediction-item">
                    <div style="flex: 1;">
                        <div class="prediction-label">${pred.class_name || `Class ${pred.class}`}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${(pred.confidence / maxConfidence) * 100}%"></div>
                        </div>
                    </div>
                    <div class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</div>
                </div>
            `).join('')}
            ${topPred ? `
                <div class="top-prediction">
                    <div class="top-prediction-label">Top Prediction</div>
                    <div class="top-prediction-value">${topPred.class_name || `Class ${topPred.class}`}</div>
                    <div style="color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.875rem;">
                        Confidence: ${(topPred.confidence * 100).toFixed(2)}%
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function displaySingleResult(result) {
    const container = document.getElementById('prediction-results');
    if (!container) return;
    
    const topPred = result.top_prediction;
    const maxConfidence = Math.max(...result.predictions.map(p => p.confidence));
    
    container.innerHTML = `
        <div class="model-card">
            <div class="model-card-header">
                <div class="model-card-title">Predictions</div>
                <div class="inference-time">
                    <i class="fas fa-clock"></i> ${result.inference_time}
                </div>
            </div>
            ${result.predictions.map((pred, idx) => `
                <div class="prediction-item">
                    <div style="flex: 1;">
                        <div class="prediction-label">${pred.class_name || `Class ${pred.class}`}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${(pred.confidence / maxConfidence) * 100}%"></div>
                        </div>
                    </div>
                    <div class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</div>
                </div>
            `).join('')}
            ${topPred ? `
                <div class="top-prediction">
                    <div class="top-prediction-label">Top Prediction</div>
                    <div class="top-prediction-value">${topPred.class_name || `Class ${topPred.class}`}</div>
                    <div style="color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.875rem;">
                        Confidence: ${(topPred.confidence * 100).toFixed(2)}%
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    // Animate confidence bars
    setTimeout(() => {
        const bars = container.querySelectorAll('.confidence-fill');
        bars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 100);
}

// Helper
function getModelName(modelId) {
    const names = {
        '12layer_cnn': '12 Layer CNN',
        'resnet50': 'ResNet50',
        '5layer_cnn': '5 Layer CNN',
    };
    return names[modelId] || modelId;
}
