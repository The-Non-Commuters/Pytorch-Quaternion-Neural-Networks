##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import init

from .ops import *


class QuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.out_channels, self.in_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                                                      self.j_weight, self.k_weight, self.bias, self.stride,
                                                      self.padding,
                                                      self.output_padding, self.groups, self.dilatation,
                                                      self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                             self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                                             self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', dilatation=' + str(self.dilatation) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', operation=' + str(self.operation) + ')'


class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.in_channels, self.out_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_conv_rotation(input, self.r_weight, self.i_weight, self.j_weight,
                                            self.k_weight, self.bias, self.stride, self.padding, self.groups,
                                            self.dilatation,
                                            self.quaternion_format)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                   self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', dilatation=' + str(self.dilatation) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', operation=' + str(self.operation) + ')'


##########

class QBatchNorm2d(Module):
    """
    Quaternion batch normalization 2d
    """

    def __init__(self,
                 in_channels,
                 affine=True,
                 training=True,
                 eps=1e-5,
                 momentum=0.9,
                 track_running_stats=True):
        """
        @type in_channels: int
        @type affine: bool
        @type training: bool
        @type eps: float
        @type momentum: float
        @type track_running_stats: bool
        """
        super(QBatchNorm2d, self).__init__()
        self.in_channels = in_channels // 4

        self.affine = affine
        self.training = training
        self.track_running_stats = track_running_stats
        self.register_buffer('eye', torch.diag(torch.cat([torch.Tensor([eps])]*4)).unsqueeze(0))

        if self.affine:
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, self.in_channels))
            self.bias = torch.nn.Parameter(torch.zeros(4, self.in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(4, self.in_channels))
            self.register_buffer('running_cov', torch.zeros(self.in_channels, 4, 4))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_cov', None)

        self.momentum = momentum

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[0, 0], 0.5)
            init.constant_(self.weight[1, 1], 1)
            init.constant_(self.weight[2, 2], 1)
            init.constant_(self.weight[3, 3], 1)

    def forward(self, x):
        x = torch.stack(torch.chunk(x, 4, 1), 1).permute(1, 0, 2, 3, 4)
        axes, d = (1, *range(3, x.dim())), x.shape[0]
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))

        if self.training:
            mean = x.mean(dim=axes)
            if self.running_mean is not None:
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean +\
                                        (1.0 - self.momentum) * mean
        else:
            mean = self.running_mean

        x = x - mean.reshape(d, *shape)

        if self.training:
            perm = x.permute(2, 0, *axes).flatten(2, -1)
            cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]

            if self.running_cov is not None:
                with torch.no_grad():
                    self.running_cov = self.momentum * self.running_cov +\
                                             (1.0 - self.momentum) * cov

        else:
            cov = self.running_cov

        ell = torch.linalg.cholesky(cov + self.eye)

        soln = torch.triangular_solve(
            x.unsqueeze(-1).permute(*range(1, x.dim()), 0, -1),
            ell.reshape(*shape, d, d),
            upper=False
        )

        invsq_cov = soln.solution.squeeze(-1)

        z = torch.stack(torch.unbind(invsq_cov, dim=-1), dim=0)

        if self.affine:
            weight = self.weight.view(4, 4, *shape)
            scaled = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1] + z[2] * weight[0, 2] + z[3] * weight[0, 3],
                z[0] * weight[1, 0] + z[1] * weight[1, 1] + z[2] * weight[1, 2] + z[3] * weight[1, 3],
                z[0] * weight[2, 0] + z[1] * weight[2, 1] + z[2] * weight[2, 2] + z[3] * weight[2, 3],
                z[0] * weight[3, 0] + z[1] * weight[3, 1] + z[2] * weight[3, 2] + z[3] * weight[3, 3],
            ], dim=0)
            z = scaled + self.bias.reshape(4, *shape)

        z = torch.cat(torch.chunk(z, 4, 0), 2)
        for _ in range(z.dim()-4):
            z.squeeze_(0)

        return z


###########


class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, gamma_init=1., affine=True, mode="covariance"):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.eps = eps
        self.momentum = momentum
        self.gamma_init = gamma_init
        self.affine = affine
        if mode not in ("variance", "covariance"):
            raise ValueError(f"unrecognized mode: {mode}")
        self.mode = mode
        if self.affine:
            self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
            self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def reset_parameters(self):
        if self.affine:
            self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
            self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1))

    def forward(self, input):

        quat_components = torch.chunk(input, 4, dim=1)

        dims = (0, *range(2, input.dim()))

        delta_r, delta_i, delta_j, delta_k = map(lambda x: x - x.mean(dim=dims, keepdim=True), quat_components)


        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2), dim=dims, keepdim=True)

        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        if self.affine:
            # Multiply gamma (stretch scale) and add beta (shift scale)
            new_r = (self.gamma * r_normalized) + beta_components[0]
            new_i = (self.gamma * i_normalized) + beta_components[1]
            new_j = (self.gamma * j_normalized) + beta_components[2]
            new_k = (self.gamma * k_normalized) + beta_components[3]
        else:
            new_r, new_i, new_j, new_k = r_normalized, i_normalized, j_normalized, k_normalized

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'


class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                              self.bias, self.quaternion_format)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'


class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'
