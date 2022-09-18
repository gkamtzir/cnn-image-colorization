from torch.utils.data import DataLoader
from CNN.Dataset import ImageDataset
import torch.optim as optim
import torch.nn as nn
from torch import save, load
from utilities import instantiate_network
import json
import time


def train_model(color_dir, gray_dir=None, epochs=50, learning_rate=0.001, architecture=1, file_name="cnn", model=None):
    """
    Trains the model with the given examples.
    :param color_dir: The directory where the colored images are stored.
    :param gray_dir: The directory where the gray images are stored.
    :param epochs: The number of epochs.
    :param learning_rate: The learning rate.
    :param architecture: The architecture of the model.
    :param file_name: The file name prefix to be used for storing model and
    results.
    :param model: (Optional) The pre-trained model.
    """
    # Loading Dataset and creating the corresponding DataLoader.
    training_data = ImageDataset(color_dir=color_dir, gray_dir=gray_dir)
    train_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

    cnn = instantiate_network(architecture)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    running_losses = []

    # Checking if there is a pre-trained model to be loaded.
    if model is not None:
        checkpoint = load(model)
        cnn.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cnn.eval()
        cnn.train()
        initial_time = checkpoint["time"]
        running_losses = checkpoint["running_losses"]
    else:
        initial_time = 0

    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters())}")

    start = time.time()
    for epoch in range(epochs):
        epoch_running_loss = 0
        for i, data in enumerate(train_data_loader, 0):
            gray, color = data
            gray = gray.float()
            color = color.float()

            outputs = cnn(gray)

            optimizer.zero_grad()
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item()/ 2000:.3f}')
        running_losses.append(epoch_running_loss)
        if epoch % 40 == 39:
            save({
                "epoch": epoch,
                "model_state_dict": cnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_losses": running_losses
            }, f"./{file_name}_{learning_rate}_{epoch}.pt")
    print("Finished")

    results = {"losses": running_losses}

    # Store losses in json file
    with open(f"figures/training_{file_name}_{learning_rate}_losses.json", "w") as results_file:
        json.dump(results, results_file)

    save({
        "epoch": epoch,
        "model_state_dict": cnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "time": time.time() - start + initial_time,
        "running_losses": running_losses
    }, f"./{file_name}_{learning_rate}_full.pt")
