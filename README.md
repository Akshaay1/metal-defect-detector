# Metal Defect Detection System

This project implements advanced deep learning methods for detecting and classifying various types of metal surface defects. The system uses state-of-the-art techniques for accurate classification of 9 common metal defect types.

## Defect Types

The system can detect and classify the following metal surface defect types:
- Crazing
- Crack
- Inclusion
- Scratches
- Indentation
- Oil Spot
- Rolling
- Rusting
- Welding Line

## Project Structure

```
├── advanced_methods.py    # Implementation of advanced detection models
├── balance_additional.py  # Data balancing and augmentation
├── evaluate_methods.py    # Model evaluation and comparison
├── train_single_model.py  # Script to train a single selected model
├── predict.py             # Prediction script for new images
├── balanced_data/         # Directory for balanced dataset
├── models/                # Directory for saved models
├── results/               # Results and metrics
├── static/                # Visualizations and static files
│   └── visualizations/    # Generated visualization images
└── logs/                  # Training logs
```

## Available Methods

The project implements several advanced deep learning approaches for metal defect detection:

1. **ArcFace Loss**: Specialized loss function for better feature space separation
2. **Multitask Learning**: Combined classification, segmentation, and defect detection
3. **Contrastive Learning**: Self-supervised learning with SimCLR architecture
4. **U-Net Architecture**: Combined segmentation and classification using U-Net
5. **Deep Texture Network**: Specialized architecture for texture-based defect recognition

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/metal_defect_detector.git
cd metal_defect_detector

# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Prepare Data

Organize your metal defect dataset into class subdirectories:

```
balanced_data/
├── crazing/
├── crack/
├── inclusion/
└── ...
```

To balance your dataset and add augmentation:

```bash
python balance_additional.py
```

### Train Models

#### Compare All Methods

To train and compare all implemented methods:

```bash
python evaluate_methods.py
```

This will:
- Train all models on your dataset
- Generate comparison metrics and visualizations
- Save results to the `results` directory

#### Train a Single Model

To train a specific model:

```bash
python train_single_model.py --model arcface
```

Available model options: `arcface`, `multitask`, `contrastive`, `unet`, `texture`

Additional options:
```
--epochs 30           # Number of training epochs
--batch_size 32       # Batch size
--data_dir your_data  # Custom data directory
--output_dir models   # Custom output directory
```

### Make Predictions

To use a trained model for prediction:

```bash
python predict.py --model_path models/arcface_model.h5 --input test_images/
```

For multitask models, add the `--multitask` flag:
```bash
python predict.py --model_path models/unet_model.h5 --input test_images/ --multitask
```

For models using ArcFace loss, add the `--arcface` flag:
```bash
python predict.py --model_path models/arcface_model.h5 --input test_images/ --arcface
```

## Results and Visualization

The system generates comprehensive visualizations:

- Confusion matrices for each model
- Training/validation loss and accuracy curves
- Class distribution charts
- Defect segmentation maps (for multitask models)
- Comparative performance metrics

Results are saved in `static/visualizations/` and `results/` directories.

## License

[MIT License](LICENSE)

## Acknowledgements

- TensorFlow and Keras teams for the deep learning framework
- The authors of the various deep learning architectures implemented
- Contributors to the open-source packages used in this project 