"""
Script to check the number of classes in your trained models
Run: python check_model_classes.py
"""

import torch
import os

def check_model_classes(model_path, model_name):
    """Check the number of classes in a model"""
    if not os.path.exists(model_path):
        print(f"[X] {model_name}: File not found at {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Try to find num_classes in checkpoint
        num_classes = None
        
        if isinstance(checkpoint, dict):
            # Check common keys
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to infer from classifier/head layer
            if isinstance(state_dict, dict):
                # Look for classifier/head weights
                for key in state_dict.keys():
                    if 'head' in key.lower() or 'classifier' in key.lower():
                        if 'weight' in key or 'fc.weight' in key:
                            weight = state_dict[key]
                            if hasattr(weight, 'shape') and len(weight.shape) >= 1:
                                num_classes = weight.shape[0]
                                print(f"[OK] {model_name}: Found classifier head with {num_classes} classes")
                                return num_classes
        
        if num_classes:
            print(f"[OK] {model_name}: {num_classes} classes (from checkpoint metadata)")
        else:
            print(f"[WARNING] {model_name}: Could not determine number of classes")
            print(f"   Please check your training code or model configuration")
        
        return num_classes
        
    except Exception as e:
        print(f"[ERROR] {model_name}: Error loading - {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("Checking Model Classes for Diabetic Retinopathy (DR) Classification")
    print("="*70)
    print()
    
    models_to_check = [
        ("models/swin.pth", "Swin Transformer"),
        ("models/beit.pth", "BEiT"),
        ("models/vit_swin_transformer.pth", "ViT Swin Transformer"),
        ("models/5layer_cnn.pth", "5 Layer CNN"),
        ("models/resnet50.pth", "ResNet50"),
        ("models/12layer_cnn.pth", "12 Layer CNN")
    ]
    
    results = {}
    for model_path, model_name in models_to_check:
        num_classes = check_model_classes(model_path, model_name)
        if num_classes:
            results[model_name] = num_classes
    
    print()
    print("="*70)
    print("Summary:")
    print("="*70)
    if results:
        for model_name, num_classes in results.items():
            print(f"  {model_name}: {num_classes} classes")
        print()
        print("[IMPORTANT] Make sure 'num_classes' in MODEL_CONFIGS matches these values!")
        print("   Update app.py if needed:")
        print("   - If binary classification (2 classes): No DR vs DR")
        print("   - If multi-class (5 classes): No DR, Mild, Moderate, Severe, Proliferative")
    else:
        print("  No models found or could not determine classes")
    print("="*70)

