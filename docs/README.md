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

### Prerequisites
- Python 3.8 or higher
- colab
- (Optional) GPU for faster training
