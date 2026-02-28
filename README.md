1. How to Run
   
To run this project and reproduce the results, please follow these steps:

  -	Open the Notebook: Ensure you are running this notebook in a Google Colab environment (or a similar Python environment with GPU access if using PyTorch models).
  -	Install Dependencies: Run the first code cell in Section 1.1 to install all necessary libraries, including pyvi, underthesea, transformers, datasets, and accelerate.
  -	Prepare Data: Ensure the DataFinalNLP.csv file is available in the working directory (or update the path in the notebook).
  -	Execute Cells Sequentially: Run all code cells in the notebook sequentially from top to bottom. This will perform:
    -	Data loading and preprocessing (Section 1.2).
    -	Dataset splitting (Section 1.3).
    -	Model architectural definitions and training for KimCNN, BiLSTM + Attention, RCNN, Transformer Encoder, and PhoBERT + Custom Head (Sections 2.1 - 2.5).
    -	Evaluation and plotting of results for each model.
    -	Public benchmark and cross – domain evaluation with PhoBERT (Section 3).
      
2. Seed for Reproducibility
 	
Note: To obtain the accurate results as shown in the report (~79.90%), please do not change the following seed initializer function in the source code:

 ```python
import torch
import numpy as np
import random

def set_seed(seed=42):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seed(42)
```

To ensure the reproducibility of experiments, the following random seeds are utilized:

  -	random_state=42: Used in sklearn.model_selection.train_test_split for consistent data partitioning.
    
  -	torch.manual_seed(s) and np.random.seed(s): Used during the stability check (Section 3.1) with seeds 42, 123, and 2024 to ensure that model initialization and training processes are consistent across different runs. While general training loops might not explicitly set torch.manual_seed in every epoch, the initial setup and critical parts are controlled.
    
3. Environment
   
This project was developed and tested using Python 3.12 within Google Colab. Key libraries and their approximate versions include:
    
  - pandas
  -	numpy
  -	re
  -	underthesea
  -	torch (PyTorch)
  -	transformers (Hugging Face Transformers)
  -	datasets (Hugging Face Datasets)
  -	scikit – learn (sklearn)
  -	matplotlib
  -	seaborn
  -	accelerate
    
It is recommended to run this notebook in a Google Colab environment or a similar setup with GPU support for faster training of deep learning models.

4. Training Parameter Configuration (Hyperparameters)
   
To ensure the reproducibility of the research results, we use the following fixed configuration:

  -	Random Seed: 42 (Set for numpy, torch, and random).
  -	Max Sequence Length: 256 tokens.
  -	Batch Size: 16 or 32 (The current model is 32).
  -	Learning Rate: 2 × 10⁻⁵ for PhoBERT and 1 × 10⁻³ for Custom Head.
  -	Optimizer: AdamW.
  -	Epochs: 10 – 20 (In the models of this project, the epochs are 20).

5. Source Code Structure
   
  -	data/: Directory containing the train.csv and test.csv files (1,940 samples).
  -	preprocess.py: Text preprocessing (icon removal, Unicode normalization, word splitting).
  -	model.py: Defines the PhoBERT architecture + Custom Head.
  -	train.py: Contains the training flow and saves the best model. 
  -	eval.py: Evaluates the model on the test set and plots the confusion matrix
