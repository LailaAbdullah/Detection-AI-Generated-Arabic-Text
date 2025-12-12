import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(cm, title, cmap='Blues', save_as=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


    if save_as is not None:
        out_path = f"/content/Detection-AI-Generated-Arabic-Text/reports/figures/{save_as}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[Saved Figure] {out_path}")


    plt.show()
    plt.close()

def plot_training_history(history, save_as=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    if save_as is not None:
        out_path = f"/content/Detection-AI-Generated-Arabic-Text/reports/figures/{save_as}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[Saved Figure] {out_path}")
        
    plt.show()
    plt.close()
