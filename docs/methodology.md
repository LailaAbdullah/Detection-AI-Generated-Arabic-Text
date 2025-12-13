# Methodology

## 1) Dataset and Initial Exploration
The dataset was downloaded and its structure examined to ensure its readiness for experimentation. This was done through:
- Reviewing the data structure (number of rows and columns) and identifying column names and data types.

- Examining the target variable (label) distribution to determine class balance (Human vs. AI-generated) and to ascertain the presence or absence of class imbalance.

- Evaluating data quality by:

- Detecting missing values ​​in text and tag columns.

- Detecting duplicates, whether at the text level or (text + tag).

- Identifying any inconsistencies or illogical values, such as empty text or unexpected tags.

## 2) Text Preprocessing and Normalization
A series of preprocessing steps were implemented for Arabic texts to reduce noise and standardize linguistic representation. These steps include:
- Normalization:

- Standardizing certain letters (e.g., the forms of alif) and processing hamzas and alif maqsurah according to a unified policy.

- Removing any non-Arabic characters/symbols, retaining only Arabic letters and spaces.

- Removing diacritics to improve consistency and reduce representational variations.

- Stop-word Removal: Removing stop words using a standard list of common Arabic words to reduce the number of words with little meaning.

- Stemming/Lemmatization:Applying ISRI Stemmer to trace words back to their morphological roots 

### Word Length–Frequency Distribution
The word length distribution within each text (Word Length Histogram) was calculated and converted into a numerical vector representing:
- The proportions/frequencies of words of different lengths (e.g., 1, 10, and 11+).

This type of feature helps capture writing patterns in terms of word length and distribution.

### Average Number of Sentences per Paragraph
The text was divided into paragraphs and then into sentences (using appropriate punctuation), and then the following was calculated:
- The average number of sentences per paragraph.

This feature reflects the text's organization and sentence distribution across paragraphs.

 ### Number of words found in the 50 positions within word embedding
A set of 50 reference words (Top-50) associated with word embeddings and consistent with the model settings used was identified.

### Perplexity score (Feature 78)
The Perplexity** score was calculated for each text using a Language Model. This score measures:
- The predictability of the text relative to the Language Model.

Texts with a more “typical” structure are generally less perplexity, which can help distinguish between human-generated and generated text.

### GPT-2 Output Probability (Feature 99)
The **probability of GPT-2 outputs** for the text was estimated by calculating the average/total probabilities of the tokens assigned to the text by GPT-2.

This feature reflects how well the text “fits” with the GPT-2 probability distribution.

## Data Splitting and Experimental Setup
The data was split into three sets to ensure fair evaluation:
- Training set
- Validation set (Hyperparameter Tuning)
- Test set (Held-out)
The value of random_state was fixed to ensure reproducibility.

## 5) Modeling

### Baseline Model 
A simple model was trained as a performance benchmark, such as Logistic Regression 
- Support Vector Machine (SVM)
- Random Forest (or any other equivalent model).
The parameters were fine-tuned using a validation set, and the best setting was selected.(SVM)

### Neural Network with BERT Embeddings 
Semantic representations (embeddings) were extracted using a custom/Arabic-compatible BERT model, and then:
- A simple neural network (feedforward network) was trained on top of these representations.

## 6) Comprehensive Evaluation (Task 4.4)
All final-selected models were evaluated on the test set using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
A confusion matrix was also created for each model to compare performance and understand errors across categories.

## 7) Error Analysis
Error analysis was performed on the best model by:
- Examining examples where the model made incorrect predictions (False Positives/False Negatives).
