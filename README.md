# Deep Learning Model Comparison System

A comprehensive web application for comparing multiple deep learning models' predictions on image classification tasks. This is a final year project that allows users to upload images and compare predictions from four different models.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or click to upload images
- 🧠 **Multiple Models**: Support for 4 different deep learning architectures:
  - 12 Layer CNN
  - ResNet50
  - 5 Layer CNN
  - ViT Swin Transformer
- 📊 **Comparison View**: See predictions from all models side-by-side
- 📈 **Performance Metrics**: View detailed metrics for each model (accuracy, precision, recall, F1-score, inference time, parameters)
- 🎨 **Modern UI**: Professional, responsive design suitable for final year project presentation

## Project Structure

```
btp_final/
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── models/               # Directory for .h5 model files
│   ├── 12layer_cnn.h5
│   ├── resnet50.h5
│   ├── 5layer_cnn.h5
│   └── vit_swin_transformer.h5
└── static/               # Frontend files
    ├── index.html        # Main HTML file
    ├── styles.css        # CSS styling
    └── script.js         # JavaScript functionality
```

## Setup Instructions

### Prerequisites

**Python 3.8-3.11 is required** (TensorFlow compatibility)

If Python is not installed:
1. Download from https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation: `python --version`

### 1. Install Dependencies

**Option A: Using the installation script (Windows)**
```bash
install.bat
```

**Option B: Manual installation**
```bash
pip install -r requirements.txt
```

**If `pip` doesn't work, try:**
```bash
python -m pip install -r requirements.txt
```

**Troubleshooting:**
- If you get "pip is not recognized": Use `python -m pip` instead
- If you get permission errors: Use `pip install --user -r requirements.txt`
- For detailed troubleshooting, see `SETUP_GUIDE.md`

### 2. Add Your Model Files

Place your `.h5` model files in the `models/` directory with the following names:
- `12layer_cnn.h5`
- `resnet50.h5`
- `5layer_cnn.h5`
- `vit_swin_transformer.h5`

### 3. Update Model Paths (if needed)

If your model files have different names, update the `MODEL_PATHS` dictionary in `app.py`:

```python
MODEL_PATHS = {
    "12layer_cnn": "models/your_12layer_model.h5",
    "resnet50": "models/your_resnet50_model.h5",
    "5layer_cnn": "models/your_5layer_model.h5",
    "vit_swin_transformer": "models/your_vit_model.h5"
}
```

### 4. Update Performance Metrics (Optional)

Edit the `PERFORMANCE_METRICS` dictionary in `app.py` with your actual model metrics:

```python
PERFORMANCE_METRICS = {
    "12layer_cnn": {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.90,
        "f1_score": 0.905,
        "inference_time": "45ms",
        "parameters": "2.3M"
    },
    # ... update other models
}
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Upload Image**: Click the upload area or drag and drop an image
2. **View Comparison**: The default view shows predictions from all models
3. **Switch Models**: Use the sidebar to view individual model details and metrics
4. **Analyze Results**: Compare predictions, confidence scores, and inference times

## API Endpoints

- `GET /api/models` - Get list of available models
- `GET /api/models/<model_id>/metrics` - Get performance metrics for a model
- `POST /api/predict` - Get prediction from a single model
- `POST /api/predict/all` - Get predictions from all models

## Model Requirements

- Models should be saved in Keras/TensorFlow `.h5` format
- Models should accept input shape: `(224, 224, 3)` (RGB images)
- Models should output class probabilities (softmax output)

## Customization

### Adjusting Input Size

If your models use a different input size, update the `preprocess_image` function in `app.py`:

```python
def preprocess_image(image, target_size=(224, 224)):
    # Change target_size to match your model's input
    ...
```

### Adding More Models

1. Add model file to `models/` directory
2. Update `MODEL_PATHS` and `MODEL_NAMES` in `app.py`
3. Add performance metrics to `PERFORMANCE_METRICS`
4. Add navigation item in `static/index.html`

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: Pillow (PIL)

## Notes

- Make sure all model files are properly loaded before making predictions
- The application assumes models output class probabilities
- Update the number of classes based on your specific classification task

## License

This project is created for educational purposes as part of a final year project.

