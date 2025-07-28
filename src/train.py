import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import copy

import src.config as config
from src.model import Network, Genotype
from src.utils import load_genotype, accuracy, plot_confusion_matrix


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("--- Loading Data and Genotype ---")
    genotype_path = os.path.join('reports', 'genotype.json')
    if not os.path.exists(genotype_path):
        print(f"Genotype file not found at {genotype_path}. Please run search.py first.")
        return

    genotype_dict = load_genotype(genotype_path)
    genotype = Genotype(
        normal=genotype_dict['normal'], normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'], reduce_concat=genotype_dict['reduce_concat']
    )

    processed_df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    X = processed_df[config.FEATURES]
    y = processed_df[config.TARGET_VARIABLE]

    X_seq, y_seq = [], []
    for i in range(len(X) - config.SEQUENCE_LENGTH):
        X_seq.append(X.iloc[i:i + config.SEQUENCE_LENGTH].values)
        y_seq.append(y.iloc[i + config.SEQUENCE_LENGTH])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    if config.USE_EARLY_STOPPING:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, shuffle=False
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    X_train = torch.from_numpy(X_train).float().permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float().permute(0, 2, 1)
    y_test = torch.from_numpy(y_test).long()

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_queue = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    test_queue = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    if config.USE_EARLY_STOPPING:
        X_val = torch.from_numpy(X_val).float().permute(0, 2, 1)
        y_val = torch.from_numpy(y_val).long()
        val_data = TensorDataset(X_val, y_val)
        val_queue = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    model = Network(C=config.INIT_CHANNELS, num_classes=2, layers=config.N_CELLS, genotype=genotype,
                    dropout_p=config.DROPOUT_RATE).cuda()

    loss_weights = None
    if config.USE_WEIGHTED_LOSS:
        print("--- Using Weighted Loss for Class Imbalance ---")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)

        if class_weights[0] > class_weights[1]:
            class_weights[0] *= config.MINORITY_CLASS_WEIGHT_MULTIPLIER
        else:
            class_weights[1] *= config.MINORITY_CLASS_WEIGHT_MULTIPLIER

        loss_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
        print(f"Final Class Weights (with multiplier): {loss_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=loss_weights).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.FINAL_LEARNING_RATE,
        weight_decay=config.FINAL_WEIGHT_DECAY
    )

    patience = config.EARLY_STOPPING_PATIENCE
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = None

    print(f"--- Starting Final Model Training (Early Stopping: {config.USE_EARLY_STOPPING}) ---")
    for epoch in range(config.FINAL_EPOCHS):
        model.train()
        train_acc, train_loss = 0, 0
        for step, (input, target) in enumerate(train_queue):
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_acc += accuracy(logits, target)
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_queue)
        avg_train_acc = train_acc / len(train_queue)

        if config.USE_EARLY_STOPPING:
            model.eval()
            val_acc, val_loss = 0, 0
            with torch.no_grad():
                for input, target in val_queue:
                    input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    logits = model(input)
                    loss = criterion(logits, target)
                    val_acc += accuracy(logits, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_queue)
            avg_val_acc = val_acc / len(val_queue)
            print(
                f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        else:
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}")

    if config.USE_EARLY_STOPPING and best_model_wts:
        print("\nLoading best model weights for final evaluation.")
        model.load_state_dict(best_model_wts)

    print("\n--- Evaluating Final Model on Test Data ---")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input, target in test_queue:
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            logits = model(input)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Down/Same', 'Up']))
    plot_confusion_matrix(all_preds, all_targets, classes=['Down/Same', 'Up'])


if __name__ == '__main__':
    main()
