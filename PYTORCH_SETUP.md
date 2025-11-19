# PyTorch Model Setup Guide

## Installation

Install PyTorch and torchvision:

```bash
pip install torch torchvision
```

For CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (CUDA), visit: https://pytorch.org/get-started/locally/

## Model Architecture Setup

**IMPORTANT**: You need to update `model_architectures.py` to match your actual model architecture!

### Steps:

1. **Check your model structure**: 
   - How many classes does your model predict?
   - What is the exact architecture (layers, channels, etc.)?

2. **Update `model_architectures.py`**:
   - Open `model_architectures.py`
   - Find the function for your model (e.g., `get_model_architecture` for "5layer_cnn")
   - Update the class definition to match your actual model architecture
   - Make sure the layer names and structure match what's in your `.pth` file

3. **Check your .pth file**:
   - Your `.pth` file should be in the `models/` folder
   - It should be named `5layer_cnn.pth` (or update `MODEL_PATHS` in `app.py`)

## Example: Updating Model Architecture

If your 5-layer CNN has a different structure, update it like this:

```python
def get_model_architecture(model_key, num_classes=None):
    if model_key == "5layer_cnn":
        class FiveLayerCNN(nn.Module):
            def __init__(self, num_classes=10):  # Update num_classes!
                super(FiveLayerCNN, self).__init__()
                # Update these layers to match your actual model
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                # ... add all your layers here ...
                self.fc = nn.Linear(512, num_classes)  # Update input size!
                
            def forward(self, x):
                # Update forward pass to match your model
                x = self.conv1(x)
                # ... rest of forward pass ...
                return self.fc(x)
        
        return FiveLayerCNN(num_classes=num_classes if num_classes else 10)
```

## Troubleshooting

### Error: "Unexpected key(s) in state_dict"
- Your model architecture doesn't match the saved state dict
- Check the layer names in your model match the keys in the `.pth` file
- You can inspect the keys with:
  ```python
  checkpoint = torch.load('models/5layer_cnn.pth', map_location='cpu')
  print(checkpoint.keys() if isinstance(checkpoint, dict) else 'Direct state dict')
  ```

### Error: "Missing key(s) in state_dict"
- Some layers in your architecture aren't in the saved model
- Make sure your model definition matches what was saved

### Model always predicts class 0
- Check if your model architecture is correct
- Verify preprocessing matches training (normalization, image size)
- Check if num_classes is correct

## Testing

Run the app and check the console output. It will show:
- Model loading status
- Prediction values
- Any errors

If you see errors, share them and I can help fix the architecture definition.

