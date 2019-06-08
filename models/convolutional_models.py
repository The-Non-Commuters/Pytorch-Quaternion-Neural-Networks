##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from torch import nn

from core.quaternion_layers import *

#
# Models are intended to work with clients in /clients
# Please use quaternion_layers.py for building custom architectures
#


class QCAE(nn.Module):  # Quaternion Convolutional AutoEncoder

    def __init__(self):
        super(QCAE, self).__init__()

        self.act = nn.Hardtanh()
        self.output_act = nn.Hardtanh()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)

        # DECODER
        self.d1 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2 = QuaternionTransposeConv(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        e1 = self.act(self.e1(x))
        e2 = self.act(self.e2(e1))

        d1 = self.act(self.d1(e2))
        d2 = self.d2(d1)

        out = self.output_act(d2)

        return out

    def network_type(self):
        return type(self).__name__


class CAE(nn.Module):  # Convolutional AutoEncoder

    def __init__(self):
        super(CAE, self).__init__()

        self.act = nn.Hardtanh()
        self.output_act = nn.Hardtanh()

        # ENCODER
        self.e1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1)

        # DECODER
        self.d1 = nn.ConvTranspose2d(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        e1 = self.act(self.e1(x))
        e2 = self.act(self.e2(e1))

        d1 = self.act(self.d1(e2))
        d2 = self.d2(d1)

        out = self.output_act(d2)

        return out

    def network_type(self):
        return type(self).__name__


class MNISTQConvNet(nn.Module):  # Quaternion CNN for MNIST

    def __init__(self):
        super(MNISTQConvNet, self).__init__()

        self.act_fn = F.relu

        self.conv1 = QuaternionConv(4, 16, kernel_size=5, stride=1, padding=1)
        self.conv2 = QuaternionConv(16, 32, kernel_size=5, stride=1, padding=1)
        self.fc1 = QuaternionLinear(800, 40)

    def forward(self, x):
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.act_fn(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 800)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1, 10, 4))
        x = torch.sum(torch.abs(x), dim=2)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class MNISTQConvNetBN(nn.Module):  # Quaternion CNN for MNIST

    def __init__(self, use_qbn):
        super(MNISTQConvNetBN, self).__init__()

        self.act_fn = F.relu
        self.use_qbn = use_qbn

        self.bn1 = QuaternionBatchNorm2d(4) if self.use_qbn else nn.BatchNorm2d(4)
        self.conv2 = QuaternionConv(4, 16, kernel_size=5, stride=1, padding=1)
        self.bn2 = QuaternionBatchNorm2d(16) if self.use_qbn else nn.BatchNorm2d(16)
        self.conv3 = QuaternionConv(16, 32, kernel_size=5, stride=1, padding=1)
        self.fc1 = QuaternionLinear(800, 40)

    def forward(self, x):
        x = self.bn1(x)
        x = self.act_fn(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = self.act_fn(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 800)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1, 10, 4))
        x = torch.sum(torch.abs(x), dim=2)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class MNISTConvNet(nn.Module):  # Standard CNN for MNIST

    def __init__(self):
        super(MNISTConvNet, self).__init__()

        self.act_fn = F.relu

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(200, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.act_fn(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 200)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARQConvNet(nn.Module):  # Quaternion CNN for CIFAR-10

    def __init__(self):
        super(CIFARQConvNet, self).__init__()

        self.act_fn = F.relu

        self.conv1 = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = QuaternionConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv3 = QuaternionConv(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = QuaternionConv(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_drop2 = nn.Dropout2d()
        self.conv5 = QuaternionConv(256, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = QuaternionLinear(8192, 40)

    def forward(self, x):
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.act_fn(F.max_pool2d(self.conv2_drop1(self.conv2(x)), 2))
        x = self.act_fn(self.conv3(x))
        x = self.act_fn(F.max_pool2d(self.conv4_drop2(self.conv4(x)), 2))
        x = self.act_fn(self.conv5(x))
        x = x.view(-1, 8192)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1, 10, 4))
        x = torch.sum(torch.abs(x), dim=2)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__


class CIFARQConvNetBN(nn.Module):  # Quaternion CNN for CIFAR-10

    def __init__(self, use_qbn):
        super(CIFARQConvNetBN, self).__init__()

        self.act_fn = F.relu
        self.use_qbn = use_qbn

        # self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # input
        self.bn1 = QuaternionBatchNorm2d(4) if self.use_qbn else nn.BatchNorm2d(4)
        self.conv1 = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = QuaternionBatchNorm2d(32) if self.use_qbn else nn.BatchNorm2d(32)
        self.conv2 = QuaternionConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.bn3 = QuaternionBatchNorm2d(64) if self.use_qbn else nn.BatchNorm2d(64)
        self.conv3 = QuaternionConv(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = QuaternionBatchNorm2d(128) if self.use_qbn else nn.BatchNorm2d(128)
        self.conv4 = QuaternionConv(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_drop2 = nn.Dropout2d()
        self.bn5 = QuaternionBatchNorm2d(256) if self.use_qbn else nn.BatchNorm2d(256)
        self.conv5 = QuaternionConv(256, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = QuaternionLinear(8192, 40)
        # self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.bn2(x)
        x = self.act_fn(F.max_pool2d(self.conv2_drop1(self.conv2(x)), 2))
        x = self.bn3(x)
        x = self.act_fn(self.conv3(x))
        x = self.bn4(x)
        x = self.act_fn(F.max_pool2d(self.conv4_drop2(self.conv4(x)), 2))
        x = self.bn5(x)
        x = self.act_fn(self.conv5(x))
        x = x.view(-1, 8192)
        x = self.act_fn(self.fc1(x))
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

        self.act_fn = F.relu

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d()
        self.conv3 = nn.Conv2d(16, 48, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(48, 92, kernel_size=5, stride=1, padding=1)
        self.conv4_drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(368, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.act_fn(F.max_pool2d(self.conv2_drop1(self.conv2(x)), 2))
        x = self.act_fn(self.conv3(x))
        x = self.act_fn(F.max_pool2d(self.conv4_drop2(self.conv4(x)), 2))
        x = x.view(-1, 368)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def network_type(self):
        return type(self).__name__
