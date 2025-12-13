# Detection of AI Generated Arabic Text
The project focuses on detecting AI-generated Arabic text using various machine learning ,deep learning models distinguish between human-written and AI-generated Arabic content.

## Features
- Word length-frequency distribution
- Average number of S/P
- Number of words found in the 50 positions within word
embedding (use corresponding word embedding aligned with the
used LLM model)
-  Perplexity score: Measures how predictable text is to a language
model.
- GPT-2 Output Probability: Probability assigned by GPT-2.


## objectives
- Developing multiple classification models for AI-generated Arabic text detection
- Comparing the performance of traditional machine learning models with deep learning methods
- Providing a comprehensive evaluation including key performance metrics with illustrative charts/visualizations of the results

## Dataset
### source:
The dataset is based on Arabic scientific abstracts from the KFUPM-JRCAI dataset,it contains Arabic text samples labeled as:
- Human-written(0)
- AI-generated(1)

## The Implemented models
- Logistic Regression
- Support Vector Machine
- Random Forest
- BERT multilingual embeddings

## Installation
### step 1: 
```bash
git clone https://github.com/LailaAbdullah/Detection-AI-Generated-Arabic-Text.git
cd Detection-AI-Generated-Arabic-Text
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Download fastText Model
Download from Google Drive**
 [Download cc.ar.300.bin (6.8 GB)](https://drive.google.com/file/d/1BsnuJZmfhpfFRcb7kVDbpt-A9psR2K2c/view?usp=share_link)]
## How to Run the Project

This project relies on two main files for operation:

### Data Preparation (data_preparation.py)

This file handles the data preparation and processing phase. Here, we clean the data, extract features, and prepare it for training.

### Run Modeling (run_modeling.py)

This is the main script where the actual classification models are run. Once the data is ready, we run this file to train the models, evaluate them, and generate the results and graphs.

### Quick Steps
1. First: Run `data_preparation.py` to prepare the data.

2. After preparation is complete, run `run_modeling.py` to train the models and compare the results.
### Prerequisites
- Python 3.8 or higher
- colab
- (Optional) GPU for faster training
