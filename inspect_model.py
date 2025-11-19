"""
Script to inspect your .pth model files and understand their structure
Run: python inspect_model.py
"""

import torch
import os

def inspect_pth_file(filepath):
    """Inspect a .pth file to understand its structure"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*70}\n")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
            
            # Check for common keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"\n✓ Found 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n✓ Found 'state_dict'")
            elif 'model' in checkpoint:
                print(f"\n✓ Found 'model' object")
                if hasattr(checkpoint['model'], 'state_dict'):
                    state_dict = checkpoint['model'].state_dict()
                else:
                    state_dict = checkpoint['model']
            else:
                # Assume the whole dict is the state dict
                state_dict = checkpoint
                print(f"\n✓ Using entire dict as state_dict")
            
            # Get other info
            if 'num_classes' in checkpoint:
                print(f"num_classes: {checkpoint['num_classes']}")
            if 'epoch' in checkpoint:
                print(f"epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                print(f"accuracy: {checkpoint['accuracy']}")
        else:
            # Direct state dict
            state_dict = checkpoint
            print(f"\n✓ Direct state dict")
        
        # Analyze state dict
        if isinstance(state_dict, dict):
            print(f"\nState dict has {len(state_dict)} keys")
            print(f"\nFirst 10 keys:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"  {i+1}. {key} -> {shape}")
            
            # Look for classifier/head
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k.lower() or 'head' in k.lower()]
            if classifier_keys:
                print(f"\n✓ Classifier/Head keys found ({len(classifier_keys)}):")
                for key in classifier_keys:
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"    - {key} -> {shape}")
            else:
                print(f"\n⚠️  No classifier/head keys found!")
            
            # Check for common prefixes
            prefixes = set()
            for key in state_dict.keys():
                if '.' in key:
                    prefix = key.split('.')[0]
                    prefixes.add(prefix)
            
            if prefixes:
                print(f"\nPrefixes found: {sorted(prefixes)}")
            
            # Check output layer
            output_keys = [k for k in state_dict.keys() if 'classifier' in k or 'head' in k or 'fc' in k or 'output' in k]
            if output_keys:
                print(f"\nOutput layer keys ({len(output_keys)}):")
                for key in output_keys:
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"    - {key} -> {shape}")
                    if hasattr(state_dict[key], 'shape') and len(state_dict[key].shape) >= 1:
                        print(f"      Output size: {state_dict[key].shape[-1]}")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Model Inspection Tool")
    print("="*70)
    
    # Check your model files
    models_to_check = [
        "models/swin.pth",
        "models/beit.pth",
        "models/5layer_cnn.pth"
    ]
    
    for model_path in models_to_check:
        if os.path.exists(model_path):
            inspect_pth_file(model_path)
        else:
            print(f"\n⚠️  File not found: {model_path}")

