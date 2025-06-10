# Social Networks Research Project

## Overview
This project is part of CS 440 Social Networks course research work, focusing on machine learning applications in social network analysis. The project uses modern deep learning frameworks and follows a structured research methodology through Jupyter notebooks. The research is conducted using the Food101 dataset, a large-scale dataset containing 101 food categories with 1000 images per category.

## Dataset
The project utilizes the Food101 dataset, which consists of:
- 101 food categories
- 1000 images per category (101,000 images total)
- High-quality food images from various sources
- Standard train/test split (75,750 training images, 25,250 test images)

The dataset is commonly used for:
- Food image classification
- Transfer learning experiments
- Computer vision research
- Deep learning model evaluation

## Project Structure
```
.
├── notebooks/               # Jupyter notebooks containing research work
│   ├── 01_data_exploration.ipynb    # Initial data analysis and exploration
│   ├── 02_preprocessing.ipynb       # Data preprocessing and feature engineering
│   └── 03_model_architecture.ipynb  # Model design and implementation
├── models/                  # Directory for saved models and model-related code
├── requirements.txt         # Python dependencies for Mac
└── requirements_windows_gpu.txt  # Python dependencies for Windows with GPU support
```

## Research Methodology
The project follows a structured research approach:

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Initial analysis of social network data
   - Statistical analysis and visualization
   - Feature identification and understanding

2. **Data Preprocessing** (`02_preprocessing.ipynb`)
   - Data cleaning and normalization
   - Feature engineering
   - Dataset preparation for model training

3. **Model Architecture** (`03_model_architecture.ipynb`)
   - Model design and implementation
   - Training pipeline development
   - Performance evaluation

## Technical Stack

### Core Dependencies
- **Deep Learning Frameworks**
  - TensorFlow (with Metal support for Apple Silicon)
  - PyTorch
  - torchvision

### Data Processing & Analysis
- pandas
- numpy
- scikit-learn

### Visualization
- matplotlib
- seaborn

### Image Processing
- albumentations
- opencv-python

### Development Tools
- Jupyter Notebook
- Weights & Biases (wandb) for experiment tracking

## Setup Instructions

### For Mac Users (Apple Silicon)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### For Windows Users (with GPU)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements_windows_gpu.txt
   ```

## Usage
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate through the notebooks in order:
   - Start with `01_data_exploration.ipynb`
   - Proceed to `02_preprocessing.ipynb`
   - Finally, work with `03_model_architecture.ipynb`

## Notes
- The project is optimized for both Mac (with Apple Silicon support) and Windows systems
- GPU acceleration is supported on Windows through the alternative requirements file
- Experiment tracking is available through Weights & Biases (optional)

## Contributing
This is a research project for CS 440 Social Networks. Please refer to course guidelines for contribution policies.

## License
Please refer to course guidelines for licensing information. 