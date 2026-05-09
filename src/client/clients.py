import copy
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.resnet import SplitModel


class BaseClient:
    def __init__(self, client_id, model, train_loader, test_loader, device, lr=0.001, local_epochs=5):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.num_samples = len(train_loader.dataset)

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return (correct / total if total > 0 else 0.0, total_loss / total if total > 0 else 0.0, total)


class StandardClient(BaseClient):
    def __init__(self, *args, mu=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def train(self, global_params):
        self.model.load_state_dict(global_params)
        global_copy = copy.deepcopy(global_params)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                if self.mu > 0:
                    prox = sum(((p - global_copy[n].to(self.device)) ** 2).sum()
                               for n, p in self.model.named_parameters())
                    loss += (self.mu / 2.0) * prox
                loss.backward()
                optimizer.step()
        return copy.deepcopy(self.model.state_dict())


class ScaffoldClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_local = {n: torch.zeros_like(p).to(self.device)
                        for n, p in self.model.named_parameters()}

    def train(self, global_params, c_global):
        self.model.load_state_dict(global_params)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        total_steps = 0
        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad.data += c_global[name].to(self.device) - self.c_local[name].to(self.device)
                optimizer.step()
                total_steps += 1
        updated_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
        c_new, c_delta = {}, {}
        for name in self.c_local:
            c_new[name] = (self.c_local[name].cpu() - c_global[name].cpu()
                          + (global_params[name].cpu() - updated_params[name].cpu()) / (total_steps * self.lr))
            c_delta[name] = c_new[name] - self.c_local[name].cpu()
        self.c_local = c_new
        return updated_params, c_delta


class FedCPAClient(BaseClient):
    def __init__(self, client_id, model, train_loader, test_loader, device,
                 lr=0.001, head_lr=0.0005, local_epochs=5, head_finetune_epochs=2):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.head_lr = head_lr
        self.local_epochs = local_epochs
        self.head_finetune_epochs = head_finetune_epochs
        self.num_samples = len(train_loader.dataset)

    def train(self, extractor_params):
        self.model.set_extractor_params(extractor_params)
        self.model.train()
        optimizer = torch.optim.Adam([
            {"params": self.model.extractor.parameters(), "lr": self.lr},
            {"params": self.model.head.parameters(), "lr": self.head_lr},
        ])
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        for p in self.model.extractor.parameters():
            p.requires_grad = False
        head_opt = torch.optim.Adam(self.model.head.parameters(), lr=self.head_lr)
        for _ in range(self.head_finetune_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                head_opt.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                head_opt.step()
        for p in self.model.extractor.parameters():
            p.requires_grad = True
        return copy.deepcopy(self.model.get_extractor_params())

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return (correct / total if total > 0 else 0.0, total_loss / total if total > 0 else 0.0, total)


class FedPerClient(FedCPAClient):
    """Same as CLADE client but without clustering."""
    pass
