from distutils.log import Log
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score

# Data preprocessing
def load_data(dataset=load_breast_cancer()):
    data = dataset
    X = data.get('data')
    y = data.get('target')
    return X, y

def profile_data(X,y):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    label_balance = y[y==1].shape[0] / y.shape[0]

    print(f"""
    \nDATA PROFILE
    Samples: {n_samples}
    Features: {n_features}
    % Class 1: {label_balance:.1%}
    """)

def preprocess(X_train, X_test, y_train, y_test, to_torch=True):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if to_torch:
        # To torch tensors
        X_train = torch.from_numpy(X_train.astype(np.float32))
        y_train = torch.from_numpy(y_train).view(y_train.shape[0],1)
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_test = torch.from_numpy(y_test).view(y_test.shape[0],1)

    return X_train, X_test, y_train, y_test

def main(X_train, X_test, y_train, y_test):
    # Model
    class LogisticRegression(nn.Module):
        def __init__(self, x: torch.Tensor) -> None:
            super().__init__()
            self.in_features = x.shape[1]
            self.logits = nn.Sequential(
                nn.Linear(self.in_features, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            logits = self.logits(x)
            return logits

    model = LogisticRegression(X_train)
    print("\nMODEL ARCHITECTURE")
    print(model)
    # Loss and Optimizer
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(params=model.parameters())

    # Training loop
    epochs = range(400)
    print("\nTRAINING...")
    for epoch in epochs:
        # do a forward pass
        y_pred = model(X_train)
        # compute loss
        loss = loss_fn(y_pred, y_train.float())

        # backwards pass (gradient descent)
        loss.backward()

        # update weights and bias
        optim.step()

        # set grad to zero
        optim.zero_grad()

        if epoch % 100 == 0:
            print(f"""Epoch: {epoch} || Loss: {loss:.3f}""")
        
    with torch.no_grad():
        test_pred = model(X_test).round()
        test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
        test_precision = precision_score(y_true=y_test, y_pred=test_pred)
        test_recall = recall_score(y_true=y_test, y_pred=test_pred)

        train_pred = model(X_train).round()
        train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
        train_precision = precision_score(y_true=y_train, y_pred=train_pred)
        train_recall = recall_score(y_true=y_train, y_pred=train_pred)

    print(f"""
    TRAINING RESULTS
    *****TEST***** 
    Accuracy:  {test_acc:.2%}
    Precision: {test_precision:.2%}
    Recall:    {test_recall:.2%}

    *****TRAIN***** 
    Accuracy:  {train_acc:.2%}
    Precision: {train_precision:.2%}
    Recall:    {train_recall:.2%}
    """)

    return model

if __name__ == "__main__":
    X,y = load_data()
    profile_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2
        )
    X_train, X_test, y_train, y_test = preprocess(
        X_train, X_test, y_train, y_test
        )
    model = main(
        X_train, X_test, y_train, y_test
        )
    