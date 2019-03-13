################################################################
# Goal:    Test the effectiveness of Group Sparse Regularization
#          on Quaternion Neural Networks
#
# Datasets:
#           MNIST    http://yann.lecun.com/exdb/mnist/
#           CIFAR10  https://www.cs.toronto.edu/~kriz/cifar.html
#
# Paper:   Group Sparse Regularization for Deep Neural Networks
#          https://arxiv.org/pdf/1607.00485.pdf
#
# Author:  Riccardo Vecchi
#
# Version: 1.0
################################################################

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from quaternion_layers import *

import matplotlib.pyplot as plt

import time

# PARAMETERS #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = 'CIFAR10'
use_quaternion_variant = True
plot_curve = True
debug = False
log_interval = 10

# HYPER PARAMETERS #
n_epochs = 1
learning_rate = 0.001
loss_criterion = F.cross_entropy  # before F.nll_loss (Negative log-likelihood loss)
batch_size_train = 200
batch_size_test = 1000
regularization_factor = 0.0001
regularizer = 'L2'

'''regularizers = {
    'L1': lambda param: torch.sum(torch.abs(param)),
    'L2': lambda param: torch.sum(param ** 2),
    'Group L1': lambda param: torch.sum(np.sqrt(param.shape[0]) * torch.sqrt(torch.sum(param[0] ** 2))),
    'Sparse GL1': lambda param: regularizers['Group L1'](param) + regularizers['L1'](param)
}'''

CIFAR10_num_to_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class MNISTQConvNet(nn.Module):  # Quaternion CNN for MNIST

    def __init__(self):
        super(MNISTQConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # input
        self.conv2 = QuaternionConv(4, 8, kernel_size=5, stride=1, padding=1)
        self.conv3 = QuaternionConv(8, 16, kernel_size=5, stride=1, padding=1)
        self.conv3_drop1 = nn.Dropout2d()
        self.fc1 = QuaternionLinear(400, 40)
        # self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop1(self.conv3(x)), 2))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1, 10, 4))
        x = torch.sum(torch.abs(x), dim=2)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class MNISTConvNet(nn.Module):  # Standard CNN for MNIST

    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop1 = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARQConvNet(nn.Module):  # Quaternion CNN for CIFAR-10

    def __init__(self):
        super(CIFARQConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # input
        self.conv2 = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = QuaternionConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv4 = QuaternionConv(64, 128, kernel_size=5, stride=1, padding=1)
        self.conv5 = QuaternionConv(128, 256, kernel_size=5, stride=1, padding=1)
        self.conv5_drop2 = nn.Dropout2d()
        self.fc1 = QuaternionLinear(1024, 40)
        # self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv3(x)), 2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5_drop2(self.conv5(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1, 10, 4))
        x = torch.sum(torch.abs(x), dim=2)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARConvNet(nn.Module):  # Standard CNN for CIFAR-10

    def __init__(self):
        super(CIFARConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv3 = nn.Conv2d(16, 48, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(48, 92, kernel_size=5, stride=1, padding=1)
        self.conv4_drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(368, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4_drop2(self.conv4(x)), 2))
        x = x.view(-1, 368)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


def get_dataset():

    if dataset == 'CIFAR10':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data = torchvision.datasets.CIFAR10

    elif dataset == 'MNIST':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (
                                                        0.3081,))])  # global mean and standard deviation for MNIST
        data = torchvision.datasets.MNIST

    train_loader = torch.utils.data.DataLoader(
        data('/files/', train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        data('/files/', train=False, download=True, transform=transform), batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


def count_trainable_parameters():
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def regularization(regularization_type=None):

    reg = 0

    if regularization_type == 'L1':
        for param in network.parameters():
            reg += torch.sum(torch.abs(param))

    elif regularization_type == 'L2':
        for param in network.parameters():
            reg += torch.sum(param ** 2)

    elif regularization_type == 'Group L1':

        square_mat_sum = None

        for param in network.parameters():

            if param.dim() > 1:  # avoid biases if exist (one-dimensional arrays)
                if square_mat_sum is not None and square_mat_sum.shape != param.shape:
                    reg += torch.sum(torch.sqrt(square_mat_sum))
                    square_mat_sum = None
                square_mat_sum = param ** 2 if square_mat_sum is None else square_mat_sum + param ** 2

    elif regularization_type == 'Sparse GL1':
        reg += regularization('Group L1') + regularization('L1')

    return reg


def calculate_sparsity():
    sparsity_weights, sparsity_neurons = [], []

    for param in network.parameters():
        param = param.to('cpu').detach().numpy().round(decimals=3)

        nonzero_weights = 1 - (param.ravel().nonzero()[0].shape[0] / param.size)
        sparsity_weights.append(nonzero_weights)

        nonzero_neurons = param.sum(axis=0).nonzero()[0].shape[0]
        sparsity_neurons.append(nonzero_neurons)

    sparsity_weights = np.mean(sparsity_weights) * 100
    sparsity_neurons = np.sum(sparsity_neurons)

    return sparsity_weights, sparsity_neurons


def expand_input(data, repeat_type='vector_RGB'):  # [BATCH X CHANNELS X WIDTH X HEIGHT]

    if repeat_type == 'repeat':  # Copy the original input also for vector components (i, j, k)
        new_input = np.repeat(data, 4, axis=1)

    elif repeat_type == 'vector_zero':  # Zero-fill for vector components (i, j, k)
        new_input = torch.zeros(data.shape[0], 4, data.shape[2], data.shape[3], dtype=torch.float, device=device)
        new_input[:, :1, :, :] = data

    elif repeat_type == 'vector_RGB':  # real part to 0 and (RGB) -> (i, j, k)
        new_input = torch.zeros((data.shape[0], 4, data.shape[2], data.shape[3]), dtype=torch.float, device=device)
        new_input[:, 1:, :, :] = data

    if debug:
        print('-----------------------')
        np.set_printoptions(threshold=None)
        print(data.shape)
        print(new_input.shape)
        print(new_input[0])

    return new_input


def train():

    network.train()

    # TRAIN LOOP #
    for epoch in range(n_epochs):

        test(epoch)

        for batch_index, (data, target) in enumerate(train_set):

            if use_quaternion_variant:
                data = expand_input(data, 'vector_RGB')

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = network(data)  # Forward pass
            loss = loss_criterion(output, target) + regularization_factor * regularization(regularizer)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            if batch_index % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_index * len(data), len(train_set.dataset),
                    100. * batch_index / len(train_set), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_index * batch_size_train) + (epoch * len(train_set.dataset)))

    test(n_epochs)


def test(epoch):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_set:

            if use_quaternion_variant:
                data = expand_input(data, 'vector_RGB')

            data, target = data.to(device), target.to(device)

            output = network(data)
            test_loss += loss_criterion(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_set.dataset)
    test_losses.append(test_loss)
    test_counter.append(epoch * len(train_set.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_set.dataset),
        100. * correct / len(test_set.dataset)))


def inference(raw_image):
    image_tensor = raw_image.unsqueeze_(0).to(device)
    if use_quaternion_variant:
        image_tensor = expand_input(image_tensor)
    network.eval()
    output = network(image_tensor)
    index = torch.argmax(output).item()
    index = CIFAR10_num_to_classes[index] if dataset == 'CIFAR10' else index
    return index


def show_image(image, text_ground_truth):
    plt.tight_layout()
    plt.subplot(2, 3, 1)
    plt.xticks([])
    plt.yticks([])

    image = np.transpose(image / 2 + 0.5, (1, 2, 0)) if dataset == 'CIFAR10' else image[0]
    plt.imshow(image, cmap='gray', interpolation='nearest')

    text_ground_truth = CIFAR10_num_to_classes[text_ground_truth] if dataset == 'CIFAR10' else text_ground_truth
    plt.title('Ground Truth: {}'.format(text_ground_truth))

    plt.show()


def plot_training_curve():
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel(loss_criterion.__name__.capitalize().replace('_', ' '))
    plt.show()


print('\n*** Group Sparse Regularization Testing ***\n')

if use_quaternion_variant:
    if dataset == 'MNIST':
        network = MNISTQConvNet()
    else:
        network = CIFARQConvNet()
else:
    if dataset == 'MNIST':
        network = MNISTConvNet()
    else:
        network = CIFARConvNet()

network = network.to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

for module in network.modules():
    print('--------------')
    print(module)

print('Device used: ' + device.type)
print('Network variant: ' + network.network_type())
print('Number of trainable parameters: {}\n'.format(count_trainable_parameters()))

print('Retrieve ' + dataset + ' dataset...\n')
train_set, test_set = get_dataset()

train_counter, train_losses, test_counter, test_losses = [], [], [], []

print('\nStart training from ' + dataset + ' training set to generate the model...')
print('Epochs: ' + str(n_epochs) + '\nLearning rate: ' + str(learning_rate) + '\n')

start_time = time.time()
train()
print('Elapsed time: {:.2f} seconds\n'.format(time.time()-start_time))

weights, neurons = calculate_sparsity()
print('Checking sparsity...\nSparsity {:.2f}%\nNeurons: {}\n'.format(weights, neurons))

samples = enumerate(test_set)
batch_idx, (sample_data, sample_targets) = next(samples)

print('Evaluation of a random sample: ' + str(inference(sample_data[0])))
show_image(sample_data[0], sample_targets[0])  # Show a random image from the test set

if plot_curve:
    plot_training_curve()
