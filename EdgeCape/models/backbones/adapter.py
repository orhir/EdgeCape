
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

"""
Code is based on: https://github.com/mbanani/probe3d
"""


class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="bindepth",
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        feats = self.head(feats)
        depth = self.predict(feats)
        return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3, hr=False, swin=False):
        super().__init__()
        assert len(input_dims) == 4
        self.hr = hr
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

        if swin:
            self.scale_factor = [1, 2, 4, 4]
        else:
            self.scale_factor = [2, 2, 2, 2]

    def forward(self, features):
        """Prediction each pixel."""
        assert len(features) == 4
        feats = features.copy()
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=scale_factor) for x, scale_factor in zip(feats, self.scale_factor)]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)
        if not self.hr:
            return self.out_conv(out)
        out = interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        # out = interpolate(out, scale_factor=2)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [interpolate(feat, (h, w), mode="bilinear") for feat in feats]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear")
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv_out(feats)

def get_norm(norm, out_channels, num_norm_groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(num_norm_groups, channels),
        }[norm]
    return norm(out_channels)


def get_activation(activation):
    """
    Args:
        activation (str or callable): either one of relu, lrelu, prelu, leaky_relu,
            sigmoid, tanh, elu, selu, swish, mish; or a callable that takes a
            tensor and returns a tensor.
    Returns:
        nn.Module or None: the activation layer
    """
    if activation is None:
        return None
    if isinstance(activation, str):
        if len(activation) == 0:
            return None
        activation = {
            "relu": nn.ReLU,
            "lrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "selu": nn.SELU,
        }[activation]
    return activation()


# SCE crisscross + diags
class EfficientSpatialContextNet(nn.Module):
    def __init__(self, kernel_size=7, in_channels=768, out_channels=768, use_cuda=True):
        super(EfficientSpatialContextNet, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            in_channels + 4 * self.kernel_size,
            out_channels,
            1,
            bias=True,
            padding_mode="zeros",
        )

        if use_cuda:
            self.conv = self.conv.cuda()

    def forward(self, feature):
        b, c, h, w = feature.size()
        feature_normalized = F.normalize(feature, p=2, dim=1)
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [4 * self.kernel_size, b, h, w],
            dtype=feature.dtype,
            requires_grad=feature.requires_grad,
        )
        if feature.is_cuda:
            output = output.cuda(feature.get_device())

        # left-top to right-bottom
        for i in range(self.kernel_size):
            c = i
            r = i
            output[i] = (feature_pad[:, :, r: (h + r), c: (w + c)] * feature_normalized).sum(1)

        # col
        for i in range(self.kernel_size):
            c = self.kernel_size // 2
            r = i
            output[1 * self.kernel_size + i] = (feature_pad[:, :, r: (h + r), c: (w + c)] * feature_normalized).sum(1)

        # right-top to left-bottom
        for i in range(self.kernel_size):
            c = (self.kernel_size - 1) - i
            r = i
            output[2 * self.kernel_size + i] = (feature_pad[:, :, r: (h + r), c: (w + c)] * feature_normalized).sum(1)

        # row
        for i in range(self.kernel_size):
            c = i
            r = self.kernel_size // 2
            output[3 * self.kernel_size + i] = (feature_pad[:, :, r: (h + r), c: (w + c)] * feature_normalized).sum(1)

        output = output.transpose(0, 1).contiguous()
        output = torch.cat((feature, output), 1)
        output = self.conv(output)
        # output = F.relu(output)

        return output


class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.
    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm="GN",
            stride_in_1x1=False,
            dilation=1,
            num_norm_groups=32,
            kernel_size=(1, 3, 1)
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels, num_norm_groups),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=kernel_size[0],
            stride=stride_1x1,
            padding=(kernel_size[0] - 1) // 2,
            bias=False,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size[1],
            stride=stride_3x3,
            padding=dilation * (kernel_size[1] - 1) // 2,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=kernel_size[2],
            bias=False,
            norm=get_norm(norm, out_channels, num_norm_groups),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class ResNet(nn.Module):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.
        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.
        Returns:
            list[CNNBlockBase]: a list of block module.
        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )
        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.
        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.
        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret

class DummyAggregationNetwork(nn.Module):  # for testing, return the input
    def __init__(self):
        super(DummyAggregationNetwork, self).__init__()
        # dummy paprameter
        self.dummy = nn.Parameter(torch.ones([]))

    def forward(self, batch, pose=None):
        return batch * self.dummy


class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """

    def __init__(
            self,
            device,
            feature_dims=[640, 1280, 1280, 768],
            projection_dim=384,
            num_norm_groups=32,
            save_timestep=[1],
            kernel_size=[1, 3, 1],
            contrastive_temp=10,
            feat_map_dropout=0.0,
    ):
        super().__init__()
        self.skip_connection = True
        self.feat_map_dropout = feat_map_dropout
        self.azimuth_embedding = None
        self.pos_embedding = None
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([]) * np.log(contrastive_temp))
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=1,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups,
                    kernel_size=kernel_size
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l + 1}")
        self.last_layer = None
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))
        # count number of parameters
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f"AggregationNetwork has {num_params} parameters.")

    def load_pretrained_weights(self, pretrained_dict):
        custom_dict = self.state_dict()

        # Handle size mismatch
        if 'mixing_weights' in custom_dict and 'mixing_weights' in pretrained_dict and custom_dict[
            'mixing_weights'].shape != pretrained_dict['mixing_weights'].shape:
            # Keep the first four weights from the pretrained model, and randomly initialize the fifth weight
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]
            custom_dict['mixing_weights'][4] = torch.zeros_like(custom_dict['mixing_weights'][4])
        else:
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]

        # Load the weights that do match
        matching_keys = {k: v for k, v in pretrained_dict.items() if k in custom_dict and k != 'mixing_weights'}
        custom_dict.update(matching_keys)

        # Now load the updated state_dict
        self.load_state_dict(custom_dict, strict=False)

    def forward(self, batch, pose=None):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        if self.feat_map_dropout > 0 and self.training:
            batch = F.dropout(batch, p=self.feat_map_dropout)

        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        if self.pos_embedding is not None:  # position embedding
            batch = torch.cat((batch, self.pos_embedding), dim=1)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature

        if self.last_layer is not None:

            output_feature_after = self.last_layer(output_feature)
            if self.skip_connection:
                # skip connection
                output_feature = output_feature + output_feature_after
        return output_feature


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)