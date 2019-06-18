##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy as np
import torch
from torch import nn

from models.convolutional_models import CAE, QCAE

from utils.misc import empty_directory, rgb2gray

import sys
import imageio

# PARAMETERS #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 3000
learning_rate = 0.0005
generation_rate = 100  # One test picture will be generated every 'generation_rate'


def main(argv):

    if len(argv) != 2:
        print('Please provide a model : QCAE or CAE')
        exit(0)

    model = str(argv[1])

    net = QCAE().to(device) if model == 'QCAE' else CAE().to(device)

    # MANAGING PICTURES #
    empty_directory('RES')

    train = rgb2gray(imageio.imread('KODAK/kodim05.png'), True)
    test = imageio.imread('KODAK/kodim23.png')

    # Normalizing
    train = train / 255
    test = test / 255

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('QCAE & CAE Color images - Titouan Parcollet - LIA, ORKIS')
    print('Model Info --------------------')
    print('Number of trainable parameters : ' + str(nb_param))

    if model == 'QCAE':
        # Add a 0 as real component if quaternions (padding)
        npad = ((0, 0), (0, 0), (1, 0))
        train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
        test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)

    # Channel first
    train = np.transpose(train, (2, 0, 1))
    test = np.transpose(test, (2, 0, 1))

    # Add batch_size dim
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

    train = torch.from_numpy(train).float().to(device)
    test = torch.from_numpy(test).float().to(device)

    for epoch in range(num_epochs):

        output = net(train)
        loss = criterion(output, train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('It : ' + str(epoch + 1) + ' | loss_train ' + str(loss.to('cpu').detach().numpy()))

        # If generation rate, generate a test image
        if (epoch % generation_rate) == 0:
            output = net(test)
            out = output.to('cpu').detach().numpy()

            out = np.transpose(out, (0, 2, 3, 1))[:, :, :, 1:] if model == 'QCAE' else np.transpose(out, (0, 2, 3, 1))
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))

            imageio.imsave('RES/save_image' + str(epoch) + '.png', out)


if __name__ == '__main__':
    main(sys.argv)
