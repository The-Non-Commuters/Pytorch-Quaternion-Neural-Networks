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

# PARAMETERS #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_quaternion_variant = True
plot_curve = True
debug = False
dataset = 'CIFAR10'
log_interval = 10

# HYPER PARAMETERS #
n_epochs = 20
learning_rate = 0.001
loss_criterion = F.cross_entropy  # before F.nll_loss (Negative log-likelihood loss)
batch_size_train = 200
batch_size_test = 1000
regularization_factor = 0.001

regularizers = {
    'L1': lambda param: torch.sum(torch.abs(param)),
    'L2': lambda param: torch.sum(param ** 2),
    # al massimo è torch.abs(param.shape[1]) e non torch.abs(param[1])
    'Group L1': lambda param: torch.sum(torch.sqrt(torch.abs(param[1])) * torch.sqrt(torch.sum(param[1] ** 2))),
    'Sparse GL1': lambda param: regularizers['Group L1'](param) + regularizers['L1'](param)
}


class MNISTQConvNet(nn.Module):  # Quaternion CNN

    def __init__(self):
        super(MNISTQConvNet, self).__init__()
        #  self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # input
        self.conv2 = QuaternionConv(4, 8, kernel_size=5, stride=1, padding=1)
        self.conv3 = QuaternionConv(8, 16, kernel_size=5, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = QuaternionLinear(400, 80)
        self.fc2 = QuaternionLinear(80, 40)

    def forward(self, x):
        #  x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        print(x.shape)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class MNISTConvNet(nn.Module):  # Standard CNN

    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARQConvNet(nn.Module):  # Quaternion CNN

    def __init__(self):
        super(CIFARQConvNet, self).__init__()
        #  self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # input
        self.conv2 = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = QuaternionConv(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv4 = QuaternionConv(64, 128, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv5 = QuaternionConv(128, 256, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2_drop2 = nn.Dropout2d()
        self.fc1 = QuaternionLinear(1024, 40, bias=False)
        # self.fc2 = QuaternionLinear(80, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        #  x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv3(x)), 2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv2_drop2(self.conv5(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        #x = x.view(-1, 4, 10)
        #print(x.shape)
        #aa = torch.zeros((x.shape[0], x.shape[2]), dtype=torch.float)
        #x = F.max_pool1d(x, 4)
        x = F.dropout(x, training=self.training)
        ## x [1000, 40]
        #x[2] = torch.argmax(x[1])
        #aa[] = x[::]
        x = self.fc2(x)
        #print(aa.shape)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARConvNet(nn.Module):  # Standard CNN

    def __init__(self):
        super(CIFARConvNet, self).__init__()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv4 = nn.Conv2d(16, 48, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(48, 92, kernel_size=5, stride=1, padding=1)
        self.conv2_drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(368, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        #  x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv3(x)), 2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv2_drop2(self.conv5(x)), 2))
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

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                         transform=transform),
            batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                         transform=transform),
            batch_size=batch_size_test, shuffle=True)

    elif dataset == 'MNIST':

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           # global mean and standard deviation of MNIST dataset
                                       ])),
            batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


def count_trainable_parameters():
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def regularization(regularization_type=None):

    reg = 0

    if regularization_type in regularizers:

        if regularization_type == 'Group L1':

            quaternion_sum = None

            for param in network.parameters():
                #print(param.shape)
                # if param.dim() > 1:  # avoid biases if exist (one-dimensional arrays)
                if quaternion_sum is None:
                    quaternion_sum = param.clone()
                elif quaternion_sum.shape != param.shape:
                    reg += torch.sum(np.sqrt(quaternion_sum.dim()) * torch.sqrt(torch.sum(quaternion_sum ** 2)))
                    quaternion_sum = param.clone()
                else:
                    quaternion_sum += param
            
        else:
            regularizer = regularizers[regularization_type]
            for param in network.parameters():
                reg += regularizer(param)

    return reg


def calculate_sparsity():
    sparsity_weights = []
    sparsity_neurons = []

    for param in network.parameters():
        nonzero_weights = 1 - (param.detach().cpu().numpy().round(decimals=2).ravel().nonzero()[0].shape[
                                   0] / count_trainable_parameters())
        sparsity_weights.append(nonzero_weights)

        nonzero_neurons = param.detach().cpu().numpy().round(decimals=2).sum(axis=0).nonzero()[0].shape[0]
        sparsity_neurons.append(nonzero_neurons)

    sparsity_weights = np.mean(sparsity_weights) * 100
    sparsity_neurons = np.sum(sparsity_neurons)

    return sparsity_weights, sparsity_neurons


def expand_input(data, repeat_type='vector_zero'):  # [BATCH X CHANNELS X WIDTH X HEIGHT]

    if repeat_type == 'repeat':  # Copy the original input also for vector components (i, j, k)
        return np.repeat(data, 4, axis=1)

    elif repeat_type == 'vector_zero':  # Zero-fill for vector components (i, j, k)
        data = np.repeat(data, 4, axis=1)
        for row in data:
            row[1].fill_(0)
            row[2].fill_(0)
            row[3].fill_(0)
        if debug:
            print('-----------------------')
            np.set_printoptions(threshold=None)
            print(data[0])
        return data

    elif repeat_type == 'vector_RGB':  # real part to 0 and (RGB) -> (i, j, k)

        '''if debug:
            print('***initial image***')
            print(data[0])

        data.resize_(data.size()[0], 4, 32, 32)  # (i, j, k, r)

        for i in range(data.size()[0]):
            data[i][3] = torch.zeros((32, 32), dtype=torch.float)
            data[i] = torch.roll(data[i], 1, 0)  # (r, i, j, k)

        if debug:
            print('-----------------------')
            np.set_printoptions(threshold=None)
            print(str(data.shape))
            print(data[0])

        return data'''

        #print(data.shape)
        new_input = torch.zeros((data.shape[0], 4, data.shape[2], data.shape[3]), dtype=torch.float, device=device)
        new_input[:, 1:, :, :] = data

        return new_input

        '''for i in range(0, batch_size_train):
            new_input[1] = data[i][0]  # i
            new_input[2] = data[i][1]  # j
            new_input[3] = data[i][2]  # k
            data[i] = new_input
        if debug:
            print('-----------------------')
            np.set_printoptions(threshold=None)
            print(str(data.shape))
            print(data[0])
        return data'''


def train():

    network.train()

    # TRAIN LOOP #
    for epoch in range(n_epochs):

        test()

        for batch_index, (data, target) in enumerate(train_set):

            if use_quaternion_variant:
                data = expand_input(data, 'vector_RGB')

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = network(data)  # Forward pass
            loss = loss_criterion(output, target) + regularization_factor * regularization('L1')
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            if batch_index % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_index * len(data), len(train_set.dataset),
                    100. * batch_index / len(train_set), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_index * batch_size_train) + (epoch * len(train_set.dataset)))

    test()


def test():
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
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_set.dataset),
        100. * correct / len(test_set.dataset)))


def inference(raw_image):
    raw_image = np.expand_dims(raw_image, axis=0)
    image_tensor = torch.from_numpy(raw_image).unsqueeze_(0)
    if use_quaternion_variant:
        image_tensor = expand_input(image_tensor).to(device)
    network.eval()
    output = network(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return index


def show_image(image, text_ground_truth):
    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.tight_layout()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title('Ground Truth: {}'.format(text_ground_truth))
    plt.xticks([])
    plt.yticks([])
    fig.show()


def plot_training_curve():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(loss_criterion.__name__.capitalize().replace('_', ' '))
    fig.show()


print('\n*** Group Sparse Regularization Testing ***')

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

print('Device used: ' + device.type)
print('Network variant: ' + network.network_type())
print('Number of trainable parameters: ' + str(count_trainable_parameters()))

optimizer = optim.Adam(network.parameters(), lr=learning_rate)

print('\nRetrieve ' + dataset + ' dataset...')
train_set, test_set = get_dataset()

train_counter = []
train_losses = []
test_counter = [i * len(train_set.dataset) for i in range(n_epochs + 1)]
test_losses = []

print('\nStart training from ' + dataset + ' training set to generate the model...')
print('Epochs: ' + str(n_epochs) + '\nLearning rate: ' + str(learning_rate) + '\n')

train()

weights, neurons = calculate_sparsity()
print('\nNeurons: {}\nSparsity {:.2f}%'.format(neurons, weights))

samples = enumerate(test_set)
batch_idx, (sample_data, sample_targets) = next(samples)

show_image(sample_data[0][0], sample_targets[0])  # Show a random image from the test set
print('Evaluation of a random sample: ' + str(inference(sample_data[0][0])))

if plot_curve:
    plot_training_curve()
