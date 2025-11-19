# Models Directory

Place your trained model files (.h5 format) in this directory with the following names:

- `12layer_cnn.h5` - 12 Layer CNN model
- `resnet50.h5` - ResNet50 model
- `5layer_cnn.h5` - 5 Layer CNN model
- `vit_swin_transformer.h5` - ViT Swin Transformer model

## Model Requirements

- Format: Keras/TensorFlow `.h5` format
- Input shape: `(224, 224, 3)` for RGB images
- Output: Class probabilities (softmax activation)

If your models have different input shapes or requirements, you may need to modify the `preprocess_image` function in `app.py`.

