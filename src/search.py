import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import src.config as config
from src.model import Network
from src.utils import save_genotype, accuracy


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.MOMENTUM
        self.network_weight_decay = args.WEIGHT_DECAY
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = self._concat(self.model.parameters()).data
        try:
            moment = self._concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        dtheta = self._concat(torch.autograd.grad(loss, self.model.parameters(),
                                                  allow_unused=True)).data + self.network_weight_decay * theta

        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        self.optimizer.zero_grad()
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / self._concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _concat(self, xs):
        return torch.cat([x.view(-1) for x in xs])


def main():
    if not torch.cuda.is_available():
        print('CUDA is not available. Exiting.')
        return

    torch.cuda.set_device(0)
    np.random.seed(2)
    torch.manual_seed(2)

    print("--- Loading and Preparing Data ---")
    processed_df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    X = processed_df[config.FEATURES]
    y = processed_df[config.TARGET_VARIABLE]

    X_seq, y_seq = [], []
    for i in range(len(X) - config.SEQUENCE_LENGTH):
        X_seq.append(X.iloc[i:i + config.SEQUENCE_LENGTH].values)
        y_seq.append(y.iloc[i + config.SEQUENCE_LENGTH])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    split_idx = int(len(X_seq) * 0.8)
    X_train_full, _ = X_seq[:split_idx], X_seq[split_idx:]
    y_train_full, _ = y_seq[:split_idx], y_seq[split_idx:]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.5, random_state=42, stratify=y_train_full
    )

    X_train = torch.from_numpy(X_train).float().permute(0, 2, 1)
    y_train = torch.from_numpy(y_train).long()
    X_valid = torch.from_numpy(X_valid).float().permute(0, 2, 1)
    y_valid = torch.from_numpy(y_valid).long()

    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)

    train_queue = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_queue = DataLoader(valid_data, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    print("--- Data Loading Complete ---")

    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(config.INIT_CHANNELS, num_classes=2, layers=config.N_CELLS, criterion=criterion).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    arch_config = type('obj', (object,), {
        'MOMENTUM': config.MOMENTUM,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'arch_learning_rate': 3e-4,
        'arch_weight_decay': 1e-3
    })()
    architect = Architect(model, arch_config)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(config.EPOCHS), eta_min=0.001)

    print("--- Starting Architecture Search ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0

        valid_iter = iter(valid_queue)
        for step, (input_train, target_train) in enumerate(train_queue):
            input_train = input_train.cuda(non_blocking=True)
            target_train = target_train.cuda(non_blocking=True)

            try:
                input_valid, target_valid = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_queue)
                input_valid, target_valid = next(valid_iter)

            input_valid = input_valid.cuda(non_blocking=True)
            target_valid = target_valid.cuda(non_blocking=True)

            architect.step(input_train, target_train, input_valid, target_valid, scheduler.get_last_lr()[0], optimizer)

            optimizer.zero_grad()
            logits = model(input_train)
            loss = criterion(logits, target_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, target_train)

        avg_loss = total_loss / len(train_queue)
        avg_acc = total_acc / len(train_queue)
        print(f'Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Accuracy = {avg_acc:.4f}')

        scheduler.step()

    genotype = model.derive_genotype()
    print("\n--- Search Complete ---")
    print("Discovered Genotype:", genotype)

    genotype_path = os.path.join('reports', 'genotype.json')
    save_genotype(genotype._asdict(), genotype_path)
    print(f"\nGenotype saved to {genotype_path}")


if __name__ == '__main__':
    main()
