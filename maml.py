import torch
import torch.nn as nn
from collections import OrderedDict
from learner import Conv_Standard
import numpy as np
import torch.nn.functional as F

class MAML(nn.Module):
    def __init__(self, args, fully_connected=1600):
        super(MAML, self).__init__()
        self.args = args
        self.learner = Conv_Standard(args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters,
                                     final_layer_size=fully_connected)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_autocoder = nn.MSELoss()

    def kemans_forward(self, datas):
        return self.learner.kmeans_forward(datas)

    def forward(self, xs, ys, xq, yq):
        create_graph = True

        if self.args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test

        fast_weights = OrderedDict(self.learner.named_parameters())

        for inner_batch in range(self.num_updates):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=self.args.train)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        if self.args.train and self.args.mix:
            query_logits, query_label_new, support_label_new, lam = self.learner.forward_metamix(xs, ys, xq, yq,
                                                                                                 fast_weights)
            query_loss = lam * self.loss_fn(query_logits, query_label_new) + (1 - lam) * self.loss_fn(query_logits,
                                                                                                      support_label_new)

            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == query_label_new).sum().float() / query_label_new.shape[0]
        else:
            query_logits = self.learner.functional_forward(xq, fast_weights)
            query_loss = self.loss_fn(query_logits, yq)

            y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
            query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc

    def forward_anil(self, xs, ys, xq, yq, dropout=0):
        create_graph = True

        if self.args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test

        logits_list = []

        inin_weights = OrderedDict(self.learner.named_parameters())
        fast_weights = OrderedDict(self.learner.logits.named_parameters())

        for inner_batch in range(self.num_updates):
            logits = self.learner.forward_anil(xs, inin_weights, fast_weights, dropout, is_training=self.args.train)
            if inner_batch == 0:
                support_logits = logits
                logits_list.append(support_logits)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        if self.args.train:
            query_logits = self.learner.forward_anil(xq, inin_weights, fast_weights, dropout, is_training=self.args.train)
            query_loss = self.loss_fn(query_logits, yq)
        else:
            with torch.no_grad():
                query_logits = self.learner.forward_anil(xq, inin_weights, fast_weights, dropout, is_training=self.args.train)
                query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        selected_rows = query_logits[0:self.args.update_batch_size]

        for i in range(15, (self.args.num_classes-1)*15+1, 15):
            selected_rows = torch.cat((selected_rows, query_logits[i:i+self.args.update_batch_size]))
        logits_list.append(selected_rows)

        return query_loss, query_acc, logits_list

    def forward_maml(self, xs, ys, xq, yq, dropout=0):
        create_graph = True

        if self.args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test
        logits_list = []
        fast_weights = OrderedDict(self.learner.named_parameters())

        del fast_weights['ca.fc.0.weight']
        del fast_weights['ca.fc.0.bias']
        del fast_weights['ca.fc.2.weight']
        del fast_weights['ca.fc.2.bias']

        for inner_batch in range(self.num_updates):
            logits = self.learner.forward_maml(xs, fast_weights, dropout, is_training=True)
            if inner_batch == 0:
                support_logits = logits
                logits_list.append(support_logits)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        if self.args.train:
            query_logits = self.learner.forward_maml(xq, fast_weights, dropout, is_training=self.args.train)
            query_loss = self.loss_fn(query_logits, yq)
        else:
            with torch.no_grad():
                query_logits = self.learner.forward_maml(xq, fast_weights, dropout, is_training=self.args.train)
                query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        selected_rows = query_logits[0:self.args.update_batch_size]

        for i in range(15, (self.args.num_classes-1)*15+1, 15):
            selected_rows = torch.cat((selected_rows, query_logits[i:i+self.args.update_batch_size]))
        logits_list.append(selected_rows)
        
        return query_loss, query_acc, logits_list

