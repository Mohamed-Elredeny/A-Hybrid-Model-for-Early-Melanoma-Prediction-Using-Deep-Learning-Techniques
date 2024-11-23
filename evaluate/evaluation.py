from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(predictions, targets):
    pred_labels = [pred["labels"] for pred in predictions]
    true_labels = [target["labels"] for target in targets]

    report = classification_report(true_labels, pred_labels)
    print(report)

    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
