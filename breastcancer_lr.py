from distutils.log import Log
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data preprocessing

data = load_breast_cancer()
X = data.get('data')
y = data.get('target')

n_samples = X.shape[0]
n_features = X.shape[1]
label_balance = y[y==1].shape[0] / y.shape[0]

print(f"""
Samples: {n_samples}
Features: {n_features}
% Class 1: {label_balance:.1%}
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# To torch tensors
X_train = torch.from_numpy(X_train_scaled.astype(np.float32))
y_train = torch.from_numpy(y_train).view(y_train.shape[0],1)
X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
y_test = torch.from_numpy(y_test).view(y_test.shape[0],1)


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
print(model)
# Loss and Optimizer
loss_fn = nn.BCELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=.05)

# Training loop
epochs = range(400)
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
        print(f"""
        Epoch: {epoch} || Loss: {loss}
        """)
    
with torch.no_grad():
    test_pred = model(X_test).round()
    test_acc = test_pred.eq(y_test).sum() / float(y_test.shape[0])

    train_pred = model(X_train).round()
    train_acc = train_pred.eq(y_train).sum() / float(y_train.shape[0])
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Train accuracy: {train_acc:.2%}")
