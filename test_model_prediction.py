"""
Test script to verify model loading and predictions
Run: python test_model_prediction.py
"""

import torch
import numpy as np
from PIL import Image
import os

# Check if timm is available
try:
    import timm
    TIMM_AVAILABLE = True
    print("[OK] timm is available")
except:
    TIMM_AVAILABLE = False
    print("[X] timm is not available")
    exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def test_model(model_path, model_name, timm_arch, num_classes=1):
    """Test loading and prediction for a model"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"[X] Model file not found: {model_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        print(f"[OK] Loaded checkpoint from {model_path}")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check head shape
        head_keys = [k for k in state_dict.keys() if 'head' in k.lower()]
        if head_keys:
            for key in head_keys:
                shape = state_dict[key].shape
                print(f"State dict {key} shape: {shape}")
        
        # Create model
        print(f"\nCreating model: {timm_arch} with num_classes={num_classes}")
        model = timm.create_model(timm_arch, pretrained=False, num_classes=num_classes)
        model.to(DEVICE)
        model.eval()
        
        # Check model head shape
        model_head_keys = [k for k in model.state_dict().keys() if 'head' in k.lower()]
        if model_head_keys:
            for key in model_head_keys:
                shape = model.state_dict()[key].shape
                print(f"Model {key} shape: {shape}")
        
        # Try loading state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[OK] Loaded state dict with strict=True")
        except Exception as e:
            print(f"[X] Strict loading failed: {e}")
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[OK] Loaded with strict=False")
                if missing:
                    print(f"  Missing keys: {missing[:5]}")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected[:5]}")
            except Exception as e2:
                print(f"[X] Failed even with strict=False: {e2}")
                return False
        
        # Create a dummy image (224x224 RGB)
        print(f"\nCreating test image (224x224 RGB)...")
        dummy_img = Image.new('RGB', (224, 224), color='red')
        
        # Preprocess
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(dummy_img).unsqueeze(0).to(DEVICE)
        print(f"Input shape: {img_tensor.shape}")
        print(f"Input range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Make prediction
        print(f"\nMaking prediction...")
        with torch.no_grad():
            outputs = model(img_tensor)
            print(f"Output shape: {outputs.shape}")
            print(f"Output values: {outputs[0].cpu().numpy()}")
            
            if outputs.shape[1] == 1:
                # Binary sigmoid
                prob_dr = torch.sigmoid(outputs[0, 0]).item()
                prob_normal = 1.0 - prob_dr
                print(f"\nBinary Classification (Sigmoid):")
                print(f"  Raw output: {outputs[0, 0].item():.6f}")
                print(f"  P(No DR): {prob_normal:.6f} ({prob_normal*100:.2f}%)")
                print(f"  P(DR): {prob_dr:.6f} ({prob_dr*100:.2f}%)")
                predicted = 1 if prob_dr > 0.5 else 0
                print(f"  Predicted: Class {predicted} ({'DR' if predicted == 1 else 'No DR'})")
            else:
                # Multi-class softmax
                probs = torch.nn.functional.softmax(outputs, dim=1)
                probs = probs[0].cpu().numpy()
                print(f"\nMulti-class Classification (Softmax):")
                for i, p in enumerate(probs):
                    print(f"  Class {i}: {p:.6f} ({p*100:.2f}%)")
                predicted = np.argmax(probs)
                print(f"  Predicted: Class {predicted}")
        
        print(f"\n[OK] Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Model Prediction Test")
    print("="*70)
    
    # Test BEiT
    test_model(
        "models/beit.pth",
        "BEiT",
        "beit_base_patch16_224",
        num_classes=1
    )
    
    # Test Swin
    test_model(
        "models/swin.pth",
        "Swin Transformer",
        "swin_base_patch4_window7_224",
        num_classes=1
    )
    
    print(f"\n{'='*70}")
    print("Test completed!")
    print("="*70)

