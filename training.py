import datetime
import time
import torch

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision import datasets
from torchvision.transforms import ToTensor

import cnn_model

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / dataset_size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss

def train_loop(epoch, dataloader, model, loss_fn, optimizer: Optimizer, writer: SummaryWriter, testDataLoader=None):
    running_loss = 0.
    last_log_ave_loss = 0.

    max_minibatch_size = dataloader.batch_size
    total_mini_batches = len(dataloader)
    total_dataset_size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for batch_idx, batch_data in enumerate(dataloader):

        inputs, labels = batch_data[0].to(cnn_model.DEVICE), batch_data[1].to(cnn_model.DEVICE)
        current_minibatch_size = len(inputs)

        pred: torch.Tensor = model(inputs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # record loss
        loss, samples_processed = loss.item(), batch_idx * max_minibatch_size + len(inputs)

        running_loss += loss

        # log loss every count of `mini_batch_loss_log_interval`
        mini_batch_loss_log_interval = 2

        if batch_idx % mini_batch_loss_log_interval == mini_batch_loss_log_interval - 1:
            print(f"epoch : {epoch + 1} | loss: {loss:>7f}  [{samples_processed:>5d}/{total_dataset_size:>5d}]")

            last_log_ave_loss = running_loss / mini_batch_loss_log_interval

            if testDataLoader != None:
                accuracy, test_loss = test_loop(testDataLoader, model, loss_fn)
                writer.add_scalars(
                    'Training Average Loss vs Validation Average Loss',
                    {'Training': last_log_ave_loss, 'Validation': test_loss},
                    epoch * total_mini_batches + batch_idx + 1
                )

                writer.add_scalar(
                    f'Validation Accuracy Every {mini_batch_loss_log_interval} Mini-batches',
                    accuracy,
                    epoch * total_mini_batches + batch_idx + 1
                )

                # Set the model to training mode - important for batch normalization and dropout layers
                model.train()

            running_loss = 0.

    writer.flush()
    return last_log_ave_loss

if __name__ == "__main__":

    # load model
    
    model = cnn_model.NeuralNetwork()
    model.to(cnn_model.DEVICE)

    # define hyperparameters

    learning_rate = 0.1
    mini_batch_size = 5000
    epochs = 30

    print(f'Hyperparameters: \nLearning Rate = {learning_rate}\nMini-Batch Size = {mini_batch_size}\nEpochs = {epochs}')

    # initialize loss/cost function and optimizers

    # Common loss functions include nn.MSELoss (Mean Square Error) for
    # regression tasks, and nn.NLLLoss (Negative Log Likelihood) for
    # classification. nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # load training and test data

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    training_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    testing_dataloader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=True)

    # per epoch training

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/training_{}'.format(timestamp))

    best_vloss = 1_000_000.

    start_time = time.time()
    epoch = 0
    while epoch < epochs:
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        ave_batch_training_loss = train_loop(
            epoch,
            training_dataloader,
            model,
            loss_function,
            optimizer,
            writer,
            testDataLoader=testing_dataloader
        )

        epoch += 1
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))