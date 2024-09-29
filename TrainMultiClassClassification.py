import datetime
import os
import sys
import time
import signal
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.backends import mps

from torchvision import datasets
from torchvision.transforms import ToTensor

from TemplateModels.MultiClassClassification import MultiClassClassification

# define hyperparameters

LEARNING_RATE = 0.001
MINI_BATCH_SIZE = 512
TRAINING_EPOCHS = 3
# MOMENTUM = None

# load training and test data

TRAIN_DATASET = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

TEST_DATASET = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=MINI_BATCH_SIZE, shuffle=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=MINI_BATCH_SIZE, shuffle=True)

# Initialize model and optimizer

MODEL = MultiClassClassification()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# Load saved model if exist for when resuming training

BEST_MODEL_SAVE_PATH = os.path.join('TemplateModels', f'best_{type(MODEL).__name__}_model.pth')
LAST_EPOCH = 0

if os.path.isfile(BEST_MODEL_SAVE_PATH):
    print('Training Checkpoint Found: Resuming Training')
    CHECKPOINT = torch.load(BEST_MODEL_SAVE_PATH, weights_only=True)
    MODEL.load_state_dict(CHECKPOINT['model_state_dict'])
    OPTIMIZER.load_state_dict(CHECKPOINT['optimizer_state_dict'])
    LAST_EPOCH = CHECKPOINT['epoch'] if CHECKPOINT['epoch'] else 0
else:
    print('Training Checkpoint Not Found: Performing Training From The Start')

# get best available compute device and load the model to it

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

MODEL.to(DEVICE)
print (f'Device Loaded To : {DEVICE}')

# initialize loss/cost function

# Common loss functions include nn.MSELoss (Mean Square Error) for
# regression tasks, and nn.NLLLoss (Negative Log Likelihood) for
# classification. nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

LOSS_FUNCTION = torch.nn.CrossEntropyLoss()

# register eventhandler to save model on exit

def save_pytorch_model():
    torch.save({
        'epoch': epoch_idx,
        'model_state_dict': MODEL.state_dict(),
        'optimizer_state_dict': OPTIMIZER.state_dict()
    }, BEST_MODEL_SAVE_PATH)

def exit_handler(sig, frame):
    # save when stopping training
    save_pytorch_model()

    print('Early Exit : Training Checkpoint Saved')
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

# defined test loop

def test_loop():
    
    # Set the MODEL to evaluation mode - important for batch normalization and dropout layers
    
    MODEL.eval()

    total_dataset_size = len(TEST_DATASET)
    total_mini_batches = len(TEST_DATALOADER)
    average_loss, correct = 0, 0

    # Evaluating the MODEL with torch.no_grad() ensures that no gradients
    # are computed during test mode also serves to reduce unnecessary gradient
    # computations and memory usage for tensors with requires_grad=True

    with torch.no_grad():
        for batch_inputs, batch_labels in TEST_DATALOADER:

            batch_inputs.to(DEVICE)
            batch_labels.to(DEVICE)

            batch_output = MODEL(batch_inputs)

            average_loss += LOSS_FUNCTION(batch_output, batch_labels).item()
            correct += (batch_output.argmax(1) == batch_labels).type(torch.float).sum().item()

    # calculate average loss and the accuracy of model during test

    average_loss /= total_mini_batches
    accuracy = correct / total_dataset_size

    print(f"test_loop  | Accuracy: {100 * accuracy:>0.2f}%, Avg. Loss: {average_loss:>8f} \n")

    return accuracy, average_loss

# main loop

if __name__ == "__main__":

    print(f'Hyperparameters: \nLearning Rate = {LEARNING_RATE}\nMini-Batch Size = {MINI_BATCH_SIZE}\nEpochs = {LAST_EPOCH + 1}/{TRAINING_EPOCHS}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/training_{}'.format(timestamp))

    # write model graph in pytorch
    writer.add_graph(MODEL, torch.rand(1, 1, 28, 28))

    best_test_loss = 1_000_000.

    # per epoch training

    start_time = time.time()
    epoch_idx = LAST_EPOCH
    
    while epoch_idx < TRAINING_EPOCHS:

        # Make sure gradient tracking is on, and do a pass over the data
        MODEL.train(True)

        # =========== TRAINING LOOP STARTS HERE =============
        
        correct = 0
        running_loss = 0.
        last_log_ave_loss = 0.

        total_dataset_size = len(TRAIN_DATASET)
        total_mini_batches = len(TRAIN_DATALOADER)

        for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):

            # Set the MODEL to training mode - important for batch normalization and dropout layers

            MODEL.train()

            # load the inputs and labels from the dataloader

            batch_inputs, batch_labels = batch_data[0].to(DEVICE), batch_data[1].to(DEVICE)

            # feed the inputs to the network

            batch_output: torch.Tensor = MODEL(batch_inputs)

            # calculate the loss of the network

            loss = LOSS_FUNCTION(batch_output, batch_labels)

            # calculate the gradients of the network and perform backpropagation

            loss.backward()
            OPTIMIZER.step()

            # zero your gradients to prevent gradient accumulation

            OPTIMIZER.zero_grad()

            # get the loss scalar value and calculate the number of processed data from the dataset

            loss, samples_processed = loss.item(), batch_idx * MINI_BATCH_SIZE + len(batch_inputs)

            # accumulate loss in each mini-batches

            running_loss += loss

            # log loss every `minibatch_log_interval`

            minibatch_log_interval = 5

            if batch_idx % minibatch_log_interval == minibatch_log_interval - 1:

                # calculate the training accuracy and loss for each minibatch log interval

                correct += (batch_output.argmax(1) == batch_labels).type(torch.float).sum().item()
                training_accuracy = correct / len(batch_labels)
                last_log_ave_loss = running_loss / minibatch_log_interval

                print(f"train_loop | Epoch : {epoch_idx + 1}/{TRAINING_EPOCHS} | Accuracy: {training_accuracy * 100:>0.2f}% | Loss: {loss:>7f} | [{samples_processed:>5d}/{total_dataset_size:>5d}]")

                # if there is a test dataloader perform validation runs and write reports to tensorboard

                if TEST_DATALOADER != None:

                    # run test loops

                    test_accuracy, test_loss = test_loop()
                    
                    # Set the MODEL back to training mode after tests
                    
                    MODEL.train()

                    # write specific reports to tensorboard

                    writer.add_scalar(
                        f'Accuracy every {minibatch_log_interval} mini-batches: Validation',
                        test_accuracy,
                        epoch_idx * total_mini_batches + batch_idx
                    )

                    writer.add_scalar(
                        f'Accuracy every {minibatch_log_interval} mini-batches: Training',
                        training_accuracy,
                        epoch_idx * total_mini_batches + batch_idx
                    )

                    writer.add_scalars(
                        f'Accuracy every {minibatch_log_interval} mini-batches: Training vs Validation',
                        {
                            'Training': training_accuracy,
                            'Validation': test_accuracy
                        },
                        epoch_idx * total_mini_batches + batch_idx
                    )

                    writer.add_scalars(
                        f'Avg. Loss per {minibatch_log_interval} mini-batches: Training vs Validation',
                        {
                            'Training': last_log_ave_loss,
                            'Validation': test_loss
                        },
                        epoch_idx * total_mini_batches + batch_idx
                    )

                    # Track best performance, and save the model's state
                    if test_loss < best_test_loss:

                        best_test_loss = test_loss
                        save_pytorch_model()

                        # another save example
                        # torch.save(model.state_dict(), best_model_saved_path)

                running_loss = 0.
                correct = 0

        writer.flush()
        epoch_idx += 1
        LAST_EPOCH = epoch_idx

        # =========== TRAINING LOOP ENDS HERE =========

    writer.close()
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
