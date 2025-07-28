import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from src.search_space import MixedOp, OPS, FactorizedReduce, ReLUConvBN
import src.config as config

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype=None):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        if genotype:
            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile(C, op_names, indices, concat, reduction)
        else:
            self.multiplier = multiplier
            self._ops = nn.ModuleList()
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, weights=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        if weights is not None:
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
            return torch.cat(states[-self._multiplier:], dim=1)
        else:
            for i in range(self._steps):
                h1 = states[self._indices[2 * i]]
                h2 = states[self._indices[2 * i + 1]]
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1)
                h2 = op2(h2)
                s = h1 + h2
                states.append(s)
            return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion=None, genotype=None, steps=4, multiplier=4, stem_multiplier=3,
                 dropout_p=0.5):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.genotype = genotype

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv1d(config.INPUT_CHANNELS, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, genotype=genotype)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(C_prev, num_classes)

        if genotype is None:
            self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.genotype is None:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
            else:
                s0, s1 = s1, cell(s0, s1)

        out = self.global_pooling(s1)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(config.PRIMITIVES)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def derive_genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if config.PRIMITIVES[k] != 'none'))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if config.PRIMITIVES[k] != 'none':
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((config.PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        return Genotype(
            normal=gene_normal, normal_concat=list(concat),
            reduce=gene_reduce, reduce_concat=list(concat)
        )
