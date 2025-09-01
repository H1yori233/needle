import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    path = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )

    block = nn.Sequential(nn.Residual(path), nn.ReLU())

    return block
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = []
    layers.append(ndl.nn.Flatten())
    layers.append(nn.Linear(dim, hidden_dim))
    layers.append(nn.ReLU())

    for _ in range(num_blocks):
        block = ResidualBlock(
            dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob
        )
        layers.append(block)

    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = ndl.nn.SoftmaxLoss()

    for batch in dataloader:
        X, y = batch
        logits = model(X)
        loss = loss_fn(logits, y)

        predicted_labels = logits.numpy().argmax(axis=1)
        correct_in_batch = (predicted_labels == y.numpy()).sum()
        total_correct += int(correct_in_batch)

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

        batch_size = X.shape[0]
        total_loss += loss.numpy() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_error = 1.0 - (total_correct / total_samples)
    return avg_error, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    train_loader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )
    test_loader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    input_dim = 28 * 28
    model = MLPResNet(dim=input_dim, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model)
        print(f"Epoch {ep+1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}")

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
