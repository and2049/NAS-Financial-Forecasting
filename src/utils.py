import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_genotype(genotype, path):
    report_dir = os.path.dirname(path)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"Created directory: {report_dir}")

    with open(path, 'w') as f:
        json.dump(genotype, f, indent=4)


def load_genotype(path):
    with open(path, 'r') as f:
        return json.load(f)


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.max(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / batch_size


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=classes, yticklabels=classes)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()

