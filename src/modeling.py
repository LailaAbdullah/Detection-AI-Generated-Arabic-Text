from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer
from tensorflow import keras
import numpy as np
import joblib
import os

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    y_test_pred = lr_model.predict(X_test)
    acc=accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, labels=[0,1], target_names=["human","ai"])
    cm = confusion_matrix(y_test, y_test_pred, labels=[0,1])

    return lr_model, acc, report, cm, y_test_pred

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    print("Training SVM")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    y_val_pred_svm = svm_model.predict(X_val)
    val_acc=accuracy_score(y_val, y_val_pred_svm)
    acc=classification_report(y_val, y_val_pred_svm)
    y_test_pred_svm = svm_model.predict(X_test)
    print("SVM Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
    report=classification_report(y_test, y_test_pred, labels=[0,1], target_names=["human","ai"])
    cm_svm = confusion_matrix(y_test, y_test_pred_svm, labels=[0,1])
    print("Confusion Matrix:\n", cm_svm)

    return svm_model,val_acc,report,cm_svm,y_test_pred_svm


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test,n_estimators=100):
    print("Training Random Forest")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)
    acc_val= accuracy_score(y_val, y_val_pred_rf)
    report_val=classification_report(y_val, y_val_pred_rf, labels=[0,1], target_names=["human","ai"])
    y_test_pred_rf = rf_model.predict(X_test)
    acc=accuracy_score(y_test, y_test_pred_rf)
    report=classification_report(y_test, y_test_pred_rf, labels=[0,1], target_names=["human","ai"])

    cm_rf = confusion_matrix(y_test, y_test_pred_rf, labels=[0,1])
    print("Confusion Matrix:\n", cm_rf)
    return rf_model,acc_val,report_val,acc,report,cm_rf,y_test_pred_rf

    

def create_bert_embeddings(train_texts, val_texts, test_texts):
     bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
     X_train_emb = bert_model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
     X_val_emb = bert_model.encode(val_texts, convert_to_numpy=True, show_progress_bar=True)
     X_test_emb = bert_model.encode(test_texts, convert_to_numpy=True, show_progress_bar=True)
     print("Train embedding shape:", X_train_emb.shape)
     return X_train_emb, X_val_emb, X_test_emb


def train_neural_network(X_train_emb, y_train, X_val_emb, y_val, X_test_emb, y_test, epochs=30):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_emb)
    X_val_s = scaler.transform(X_val_emb)
    X_test_s = scaler.transform(X_test_emb)
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print("class_weight:", class_weight)

    ffnn_model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_train_s.shape[1],),
                         kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid")
    ])
  
    ffnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc_roc"), 
                keras.metrics.AUC(name="auc_pr", curve="PR")]
    )
  
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", 
                                    patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", 
                                        factor=0.5, patience=2, min_lr=1e-6)
    ]


    history = ffnn_model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
                            epochs=epochs, batch_size=32, class_weight=class_weight,
                            callbacks=callbacks)
  
    y_prob = ffnn_model.predict(X_test_s).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    test_acc=accuracy_score(y_test, y_pred)


    report=classification_report(y_test, y_pred, labels=[0,1], target_names=["human","ai"], digits=4)
    cm=confusion_matrix(y_test, y_pred, labels=[0,1])

    return ffnn_model,history,y_pred,test_acc,report,cm


def save_models(lr_model, svm_model, rf_model, nn_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(output_dir, 'lr_model.pkl'))
    joblib.dump(svm_model, os.path.join(output_dir, 'svm.pkl'))
    joblib.dump(rf_model, os.path.join(output_dir, 'random_forest.pkl'))
    nn_model.save(os.path.join(output_dir, 'ffnn.keras'))
    print("Models saved successfully.")
