
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.visualization import plot_confusion_matrix, plot_training_history
from src.utils import load_and_split_data, prepare_features, convert_labels_to_binary
from src.modeling import (train_logistic_regression, train_svm, train_random_forest,
                        create_bert_embeddings, train_neural_network, save_models)


REPO_DIR = Path("/content/Detection-AI-Generated-Arabic-Text")
sys.path.append(str(REPO_DIR))

def main():
  (REPO_DIR / "reports" / "figures").mkdir(parents=True, exist_ok=True)
  (REPO_DIR / "models").mkdir(parents=True, exist_ok=True)
  train_df, val_df, test_df = load_and_split_data(str(REPO_DIR / "data" / "processed.csv"))

  num_cols = [
        'Total number of sentences',
        'Total number of paragraphs ',
        'Average number of S/P',
        'num_words_in_top50_embedding',
        'perplexity',
        'gpt2_output_probability'
    ]

  X_train, X_val, X_test, y_train, y_val, y_test = prepare_features(train_df, val_df, test_df, num_cols)

  #LR
  lr_model, lr_acc, lr_report, lr_cm, _ = train_logistic_regression(X_train, y_train, X_test, y_test)
  print("\n[LR] Test Accuracy:", lr_acc)
  print(lr_report)
  plot_confusion_matrix(lr_cm, "Confusion Matrix - Logistic Regression", save_as="lr_cm.png")

  #svm
  svm_model, svm_val_acc, svm_val_report, svm_cm, svm_test_pred = train_svm(
  X_train, y_train, X_val, y_val, X_test, y_test )

  svm_test_acc = accuracy_score(y_test, svm_test_pred)
  print("\n[SVM] Val Accuracy:", svm_val_acc)
  print(svm_val_report)
  print("[SVM] Test Accuracy:", svm_test_acc)
  print(classification_report(y_test, svm_test_pred))
  plot_confusion_matrix(svm_cm, "Confusion Matrix - SVM", save_as="svm_cm.png")

  #RF
  rf_model, rf_val_acc, rf_val_report, rf_test_acc, rf_test_report, rf_cm, _ = train_random_forest(
  X_train, y_train, X_val, y_val, X_test, y_test, n_estimators=200)

  print("\n[RF] Val Accuracy:", rf_val_acc)
  print(rf_val_report)
  print("[RF] Test Accuracy:", rf_test_acc)
  print(rf_test_report)
  plot_confusion_matrix(rf_cm, "Confusion Matrix - Random Forest", cmap="Purples", save_as="rf_cm.png")

  #EMB+FFNN

  X_train_emb, X_val_emb, X_test_emb = create_bert_embeddings(
        train_df["cleaned_text"].tolist(),
        val_df["cleaned_text"].tolist(),
        test_df["cleaned_text"].tolist()
    )

  y_train_bin=convert_labels_to_binary(train_df)
  y_val_bin=convert_labels_to_binary(val_df)
  y_test_bin=convert_labels_to_binary(test_df)

  print("\nDistribution of y_train:", np.bincount(y_train_bin))
  print("Distribution of y_test:", np.bincount(y_test_bin))

  nn_model, history, nn_pred, nn_acc, nn_report, nn_cm = train_neural_network(
  X_train_emb, y_train_bin,X_val_emb, y_val_bin,X_test_emb, y_test_bin,epochs=30)


  print("\n[FFNN] Test Accuracy:", nn_acc)
  print(nn_report)
  plot_confusion_matrix(nn_cm, "Confusion Matrix - FFNN", save_as="ffnn_cm.png")
  plot_training_history(history, save_as="ffnn_history.png")
  save_models(lr_model, svm_model, rf_model, nn_model, str(REPO_DIR / "models"))
  print("Models saved successfully.")

main()
