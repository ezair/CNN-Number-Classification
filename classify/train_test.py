""" Contains code for training the Dog v.s. MNIST
"""

import torch
import torch.nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def train(training_set, model, optimizer, loss_function, epochs=100):
    model = model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        num_correct = 0
        num_samples = 0
        for batch, (data, targets) in enumerate(training_set):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            # Forward propagation.
            scores = model(data)

            loss = loss_function(scores, targets)

            # Backward propagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")

            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        print(f"Training Accuracy: {num_correct / num_samples * 100:.4f}")


def test(testing_set, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in testing_set:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"{num_correct} / {num_samples} correct.\n")
        print(f"Testing Accuracy: {num_correct / num_samples * 100:.4f}")
