import torch
import typer
from data_solution import corrupt_mnist
from model_solution import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(DEVICE)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)


    statistics = {"train_loss": [], "train_accuracy": []}

    for e in range(epochs):
        epoch_loss=[]
        epoch_accuracy=[]
        for i,(image, label) in enumerate(train_dataloader):
            img,label=image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fn(pred,label)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            epoch_loss.append(loss.item())
            

            accuracy = (pred.argmax(dim=1) == label).float().mean().item()
            statistics['train_accuracy'].append(accuracy)
            epoch_accuracy.append(accuracy)


        #training_losses.append(sum(running_loss)/len(train_set))
        print(f'Epoch {e+1}: Training Loss {sum(epoch_loss)/len(epoch_loss)}. Training accuracy {sum(epoch_accuracy)/len(epoch_accuracy)}')
    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")




@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)


    statistics={'test_loss':[],'test_accuracy':[]}
    correct, total = 0, 0
    for i,(image, label) in enumerate(test_dataloader):
        img,label=image.to(DEVICE),label.to(DEVICE)

        pred = model(img)
        loss = loss_fn(pred,label)

        statistics['test_loss'].append(loss.item())
            

        # accuracy  = (pred.argmax(dim=1) == label).float().mean().item()
        # statistics['test_accuracy'].append(accuracy)
        correct += (pred.argmax(dim=1) == label).float().sum().item()
        total += label.size(0)
    print(f"Test accuracy: {correct / total}")







if __name__ == "__main__":
    app()
