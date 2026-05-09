# Facial Expression Recognition with EfficientNet-B0 and PyTorch

## Overview

This project builds a deep learning pipeline for classifying human facial expressions from face images using PyTorch. The notebook trains a transfer-learning image classifier on a facial expression dataset containing 48 x 48 pixel face images distributed across seven emotion categories: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

The implementation follows a guided computer vision workflow: load a facial expression dataset, apply image augmentation, build a pretrained convolutional neural network, train and validate the model, save the best-performing weights, and use the trained model for expression prediction on validation images.

At the core of the project is an EfficientNet-B0 model loaded through the `timm` library. The pretrained backbone is adapted for a seven-class facial expression recognition task by replacing the classifier output with seven logits. The model is then optimized with cross-entropy loss and the Adam optimizer.

## Objectives

- Load and prepare a facial expression dataset for supervised image classification.
- Understand the structure of the facial expression recognition dataset and its target labels.
- Apply image transformations and augmentation to improve model generalization.
- Use a pretrained state-of-the-art convolutional neural network as the classification backbone.
- Implement reusable training and evaluation functions in PyTorch.
- Track loss and multiclass accuracy during training and validation.
- Save the best model checkpoint based on validation loss.
- Perform inference on validation images and visualize class probabilities.

## Dataset Description

The project uses a facial expression recognition dataset originally referenced from Kaggle:

https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

Inside the notebook, the data is loaded by cloning a prepared dataset repository:

https://github.com/parth1620/Facial-Expression-Dataset

The dataset is organized into training and validation directories and loaded with `torchvision.datasets.ImageFolder`, which infers class labels from folder names. The notebook uses the following dataset paths:

| Split | Path | Number of Images |
| --- | --- | ---: |
| Training | `/content/Facial-Expression-Dataset/train/` | 28,821 |
| Validation | `/content/Facial-Expression-Dataset/validation/` | 7,066 |

The detected class-to-index mapping is:

| Class | Label Index |
| --- | ---: |
| `angry` | 0 |
| `disgust` | 1 |
| `fear` | 2 |
| `happy` | 3 |
| `neutral` | 4 |
| `sad` | 5 |
| `surprise` | 6 |

Although the source images are described as 48 x 48 grayscale face images, the loaded training batch has shape `[32, 3, 48, 48]`. This means each batch contains 32 images, represented as 3-channel tensors with height and width of 48 pixels. This channel layout is compatible with the pretrained EfficientNet-B0 backbone, which expects RGB-style image tensors.

## Technical Stack

| Component | Usage |
| --- | --- |
| Python | Notebook execution and model development |
| PyTorch | Tensor operations, neural network definition, loss computation, optimization, and GPU execution |
| Torchvision | Dataset loading with `ImageFolder`, image transforms, and `DataLoader` integration |
| timm | Loading the pretrained `efficientnet_b0` model |
| NumPy | Numeric utilities, including initialization of the best validation loss |
| Matplotlib | Image display and inference probability visualization |
| tqdm | Training and validation progress bars |
| CUDA | GPU acceleration for model training and inference |

The notebook also installs `albumentations` and `opencv-contrib-python`, but the active augmentation pipeline is implemented with `torchvision.transforms`.

## Model Architecture

The classifier is implemented as a custom PyTorch module named `FaceModel`. It wraps a pretrained EfficientNet-B0 model created with:

```python
timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)
```

EfficientNet-B0 is used as the feature extraction and classification backbone. The final classification layer is configured to produce seven output logits, one for each facial expression class.

The model forward pass supports two modes:

- When labels are provided, it returns both logits and cross-entropy loss.
- When labels are not provided, it returns logits only for inference.

This design keeps the training loop concise because the model itself computes the supervised classification loss during training and validation.

## Project Workflow

1. **Dataset acquisition**  
   The notebook clones the prepared facial expression dataset repository and references the original Kaggle dataset source.

2. **Configuration setup**  
   Training and validation paths, learning rate, batch size, epoch count, device, and model name are defined. The main configuration values are `LR = 0.001`, `BATCH_SIZE = 32`, `EPOCHS = 15`, `DEVICE = 'cuda'`, and `MODEL_NAME = 'efficientnet_b0'`.

3. **Image transformation and augmentation**  
   Training images are augmented with random horizontal flips and random rotations. Validation images are converted to tensors without augmentation.

4. **Dataset loading with `ImageFolder`**  
   The training and validation folders are loaded as image classification datasets. Folder names are automatically mapped to integer labels.

5. **Batch creation with `DataLoader`**  
   The datasets are wrapped in PyTorch data loaders. The training loader shuffles samples, while the validation loader keeps a deterministic order.

6. **Pretrained EfficientNet-B0 model construction**  
   A pretrained EfficientNet-B0 model is loaded from `timm` and configured for seven expression classes.

7. **Training and validation loop execution**  
   Custom `train_fn` and `eval_fn` functions iterate over batches, compute loss, calculate multiclass accuracy, and report progress with `tqdm`.

8. **Best-weight checkpointing**  
   After each epoch, validation loss is compared with the best loss seen so far. When validation loss improves, the model weights are saved to `best-weights.pt`.

9. **Random validation-image inference**  
   The best saved weights are loaded, a random validation image is selected, and the model predicts class logits without gradient computation.

10. **Probability visualization across expression classes**  
    Softmax converts logits into class probabilities. The notebook visualizes the input face alongside a horizontal bar chart showing predicted probabilities for all seven expression classes.

## Data Preprocessing and Augmentation

The preprocessing pipeline converts images into PyTorch tensors and prepares them for mini-batch training. The notebook uses separate transformation pipelines for training and validation.

Training transformations:

```python
T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor()
])
```

Validation transformations:

```python
T.Compose([
    T.ToTensor()
])
```

The random horizontal flip helps the model learn expression patterns that are not dependent on left-right facial orientation. The random rotation introduces tolerance to minor pose and alignment variations. Validation data is not augmented because validation metrics should estimate performance on unmodified held-out samples.

After batching, the notebook confirms the following tensor dimensions:

| Tensor | Shape |
| --- | --- |
| Image batch | `[32, 3, 48, 48]` |
| Label batch | `[32]` |

## Training and Evaluation Strategy

The training loop uses supervised multiclass classification. For each training batch, images and labels are moved to CUDA, gradients are reset, logits and loss are computed, backpropagation is performed, and the optimizer updates model parameters.

The training function performs the following operations:

- Sets the model to training mode with `model.train()`.
- Moves each image and label batch to the configured device.
- Computes logits and cross-entropy loss through the model forward pass.
- Runs `loss.backward()` to compute gradients.
- Updates model parameters with `optimizer.step()`.
- Accumulates average loss and multiclass accuracy across batches.

The evaluation function mirrors the training loop but uses `model.eval()` and does not update model parameters. It computes validation loss and validation accuracy across the validation loader.

The notebook defines multiclass accuracy by selecting the highest-probability class with `topk(1)` and comparing it against the ground-truth label:

```python
top_p, top_class = y_pred.topk(1, dim=1)
equals = top_class == y_true.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))
```

Optimization configuration:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Loss function | Cross-entropy loss |
| Epochs | `15` |
| Device | CUDA |
| Batch size | `32` |

Checkpointing is based on validation loss rather than training loss. This ensures that the saved model corresponds to the best observed generalization performance during the training run.

## Results

The model is trained for 15 epochs. The best validation checkpoint is saved whenever validation loss improves. The lowest validation loss observed in the notebook occurs at epoch 14:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Checkpoint Saved |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.891768 | 0.368455 | 1.364895 | 0.481607 | Yes |
| 2 | 1.329985 | 0.491564 | 1.191074 | 0.548599 | Yes |
| 3 | 1.214177 | 0.540970 | 1.123214 | 0.584167 | Yes |
| 4 | 1.150005 | 0.563205 | 1.104723 | 0.592608 | Yes |
| 5 | 1.110353 | 0.582544 | 1.076260 | 0.600091 | Yes |
| 6 | 1.079614 | 0.590970 | 1.055378 | 0.607172 | Yes |
| 7 | 1.045419 | 0.605671 | 1.067131 | 0.597524 | No |
| 8 | 1.021004 | 0.619601 | 1.041242 | 0.622727 | Yes |
| 9 | 0.993732 | 0.629466 | 1.051278 | 0.612513 | No |
| 10 | 0.964734 | 0.637202 | 0.996812 | 0.629612 | Yes |
| 11 | 0.938024 | 0.649255 | 0.988747 | 0.633897 | Yes |
| 12 | 0.909287 | 0.661428 | 1.019820 | 0.629340 | No |
| 13 | 0.878591 | 0.672877 | 0.978397 | 0.645623 | Yes |
| 14 | 0.855130 | 0.679222 | 0.974957 | 0.648277 | Yes |
| 15 | 0.823415 | 0.692992 | 1.018092 | 0.640304 | No |

The final epoch achieves the highest training accuracy, but epoch 14 provides the best validation loss. Because checkpointing is driven by validation loss, `best-weights.pt` corresponds to epoch 14 in this run.

## Inference Pipeline

The inference stage uses the best saved checkpoint for prediction on randomly selected validation images. The notebook repeatedly performs the same inference sequence:

1. Load the saved model weights from `best-weights.pt`.
2. Randomly select an image from the validation dataset.
3. Add a batch dimension with `image.unsqueeze(0)`.
4. Move the image tensor to CUDA.
5. Run the model in evaluation mode under `torch.no_grad()`.
6. Convert logits to probabilities using `torch.softmax(logits, dim=1)`.
7. Display the image and class probability distribution with Matplotlib.

The visualization function, `view_classify`, displays two panels: the selected face image and a horizontal probability chart for the seven expression classes. This makes the prediction interpretable because the user can inspect both the model input and the confidence assigned to each facial expression.

## Key Takeaways

- Transfer learning with EfficientNet-B0 provides a compact and effective baseline for facial expression recognition.
- `ImageFolder` is well suited for datasets organized by class-specific folders because it automatically creates the label mapping.
- Lightweight augmentation, including horizontal flips and small rotations, helps expose the model to realistic facial pose variation.
- Validation-loss checkpointing is important because the final training epoch is not necessarily the best generalization point.
- The notebook demonstrates the complete classification lifecycle: dataset loading, augmentation, model construction, training, validation, checkpointing, and visual inference.
