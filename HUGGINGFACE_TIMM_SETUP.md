# HuggingFace & timm Model Setup Guide

## Overview

The application now supports multiple ways to load models:
1. **Saved .pth files** - Your trained models
2. **timm library** - Direct loading from timm
3. **HuggingFace** - Direct loading from HuggingFace transformers

## Installation

```bash
pip install torch torchvision timm transformers
```

## Configuration

Models are configured in `app.py` in the `MODEL_CONFIGS` dictionary. For your BEiT and Swin models:

### Current Configuration

```python
"beit": {
    "type": "pth_file",  # First try to load your saved .pth file
    "source": "models/beit.pth",
    "timm_fallback": "beit_base_patch16_224",  # If .pth not found, use timm
    "huggingface_fallback": "microsoft/beit-base-patch16-224",  # Or HuggingFace
    "num_classes": 4  # Update this to match your retinal disease classes
},
"swin": {
    "type": "pth_file",
    "source": "models/swin.pth",
    "timm_fallback": "swin_base_patch4_window7_224",
    "huggingface_fallback": "microsoft/swin-base-patch4-window7-224",
    "num_classes": 4  # Update this
}
```

## How It Works

### Priority Order:
1. **First**: Tries to load your saved `.pth` file from `models/beit.pth` or `models/swin.pth`
2. **If .pth found**: Loads the state dict and tries to match it with:
   - timm architecture (if `timm_fallback` specified)
   - HuggingFace architecture (if `huggingface_fallback` specified)
3. **If .pth not found**: Falls back to loading directly from timm or HuggingFace

### Loading Your Saved Models

1. **Place your .pth files** in the `models/` folder:
   - `models/beit.pth`
   - `models/swin.pth`

2. **Update `num_classes`** in `MODEL_CONFIGS` to match your retinal disease classification:
   ```python
   "num_classes": 4  # Change to your actual number of classes
   ```

3. **Update class names** (optional) in `CLASS_NAMES` dictionary:
   ```python
   CLASS_NAMES = {
       0: "Normal",
       1: "Diabetic Retinopathy",
       2: "Glaucoma",
       3: "AMD"
   }
   ```

## Model Loading Options

### Option 1: Use Your Saved .pth Files (Recommended)
- Place your trained models in `models/` folder
- The app will automatically detect and load them
- Uses timm/HuggingFace architecture as fallback if needed

### Option 2: Load Directly from timm
Change the config to:
```python
"beit": {
    "type": "timm",
    "source": "beit_base_patch16_224",
    "num_classes": 4
}
```

### Option 3: Load Directly from HuggingFace
Change the config to:
```python
"beit": {
    "type": "huggingface",
    "source": "microsoft/beit-base-patch16-224",
    "num_classes": 4
}
```

## Common timm Model Names

For BEiT:
- `beit_base_patch16_224`
- `beit_large_patch16_224`
- `beit_base_patch16_384`

For Swin:
- `swin_base_patch4_window7_224`
- `swin_large_patch4_window7_224`
- `swin_small_patch4_window7_224`

## Common HuggingFace Model Names

For BEiT:
- `microsoft/beit-base-patch16-224`
- `microsoft/beit-large-patch16-224`

For Swin:
- `microsoft/swin-base-patch4-window7-224`
- `microsoft/swin-large-patch4-window7-224`

## Troubleshooting

### Error: "Unexpected key(s) in state_dict"
- Your saved model's state dict keys don't match the architecture
- Try a different model variant (base vs large)
- Check if you need to remove prefix like "model." from keys

### Error: "Model file not found"
- Check that your .pth files are in the `models/` folder
- Verify the file names match the config
- The app will automatically fall back to timm/HuggingFace if file not found

### Model always predicts class 0
- Check that `num_classes` matches your model
- Verify preprocessing matches training (ImageNet normalization)
- Check console output for prediction values

## Testing

1. Start the app: `python app.py`
2. Check console for model loading messages
3. Upload an image and check predictions
4. Look for any errors in the console

## Next Steps

1. **Update num_classes**: Set to your actual number of retinal disease classes
2. **Update class names**: Add your disease class names to `CLASS_NAMES`
3. **Test with your models**: Place .pth files and test predictions
4. **Adjust preprocessing**: If needed, update normalization in `preprocess_image`

