"""DNN architectures based on STRF kernels."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MLP, hilbert, nextpow2


def is_strf_param(nm):
    """Check if a parameter name string is one of STRF parameters."""
    return any(n in nm for n in ("rates_", "scales_", "phis_", "thetas_"))


class GaborSTRFConv(nn.Module):
    """Gabor-STRF-based cross-correlation kernel."""
    def __init__(self, supn, supk, nkern, rates=None, scales=None):
        """Instantiate a Gabor-based STRF convolution layer.

        Parameters
        ----------
        supn: int
            Time support in number of frames. Also the window length.
        supk: int
            Frequency support in number of channels. Also the window length.
        nkern: int
            Number of kernels, each with a learnable rate and scale.
        rates: list of float, None
            Initial values for temporal modulation.
        scales: list of float, None
            Initial values for spectral modulation.

        """
        super(GaborSTRFConv, self).__init__()
        if supk % 2 == 0:  # force odd number
            supk += 1
        self.supk = torch.arange(supk, dtype=torch.float32)
        if supn % 2 == 0:  # force odd number
            supn += 1
        self.supn = torch.arange(supn, dtype=self.supk.dtype)
        self.padding = (supn//2, supk//2)

        # Set up learnable parameters
        for param in (rates, scales):
            assert (not param) or len(param) == nkern
        if not rates:
            rates = torch.rand(nkern) * math.pi
        if not scales:
            scales = torch.rand(nkern) * math.pi
        self.rates_ = nn.Parameter(torch.Tensor(rates))
        self.scales_ = nn.Parameter(torch.Tensor(scales))

    def strfs(self):
        """Make STRFs using the current parameters."""
        if self.supn.device != self.rates_.device:  # for first run
            self.supn = self.supn.to(self.rates_.device)
            self.supk = self.supk.to(self.rates_.device)
        n0, k0 = self.padding
        nsin = torch.sin(torch.ger(self.rates_, self.supn-n0))
        ncos = torch.cos(torch.ger(self.rates_, self.supn-n0))
        ksin = torch.sin(torch.ger(self.scales_, self.supk-k0))
        kcos = torch.cos(torch.ger(self.scales_, self.supk-k0))
        nwind = .5 - .5 * torch.cos(2*math.pi*self.supn/(len(self.supn)+1))
        kwind = .5 - .5 * torch.cos(2*math.pi*self.supk/(len(self.supk)+1))
        strfr = torch.bmm((ncos*nwind).unsqueeze(-1),
                          (kcos*kwind).unsqueeze(1))
        strfi = torch.bmm((nsin*nwind).unsqueeze(-1),
                          (ksin*kwind).unsqueeze(1))

        return torch.cat((strfr, strfi), 0)

    def forward(self, sigspec):
        """Forward pass real spectra [Batch x Time x Frequency]."""
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)
        strfs = self.strfs().unsqueeze(1).type_as(sigspec)
        return F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)


class STRFConv(nn.Module):
    """Spectrotemporal receptive field (STRF)-based convolution."""
    def __init__(self, fr, bins_per_octave, suptime, supoct, nkern,
                 rates=None, scales=None, phis=None, thetas=None):
        """Instantiate a STRF convolution layer.

        Parameters
        ----------
        fr: int
            Frame rate of the incoming spectrogram in Hz.
            e.g. spectrogram with 10ms hop size has frame rate 100Hz.
        bins_per_octave: int
            Number of frequency dimensions per octave in the spectrogram.
        suptime: float
            Maximum time support in seconds.
            All kernels will span [0, suptime) seconds.
        supoct: float
            Maximum frequency support in number of octaves.
            All kernels will span [-supoct, supoct] octaves.
        nkern: int
            Number of learnable STRF kernels.
        rates: array_like, (None)
            Init. for learnable stretch factor in time.
            Dimension must match `nkern` if specified.
        scales: int or float, (None)
            Init. for learnable stretch factor in frequency.
            Dimension must match `nkern` if specified.
        phis: float, (None)
            Init. for learnable phase shift of spectral evolution in radians.
            Dimension must match `nkern` if specified.
        thetas: float, (None)
            Init. for learnable phase shift of time evolution in radians.
            Dimension must match `nkern` if specified.

        """
        super(STRFConv, self).__init__()

        # For printing
        self.__rep = f"""STRF(fr={fr}, bins_per_octave={bins_per_octave},
            suptime={suptime}, supoct={supoct}, nkern={nkern},
            rates={rates}, scales={scales}, phis={phis},
            thetas={thetas})"""

        # Determine time & frequency support
        _fsteps = int(supoct * bins_per_octave)  # spectral step on one side
        self.supf = torch.linspace(-supoct, supoct, steps=2*_fsteps+1)
        _tsteps = int(fr*suptime)
        if _tsteps % 2 == 0:  # force odd number
            _tsteps += 1
        self.supt = torch.arange(_tsteps).type_as(self.supf)/fr
        self.padding = (_tsteps//2, _fsteps)
        self.ndft = max(nextpow2(max(len(self.supf), len(self.supt))), 128)

        # Set up learnable parameters
        for param in (rates, scales, phis, thetas):
            assert (not param) or len(param) == nkern
        if not rates:
            rates = torch.rand(nkern) * 10
        if not scales:
            scales = torch.rand(nkern) / 5
        if not phis:
            phis = 2*math.pi * torch.rand(nkern)
        if not thetas:
            thetas = 2*math.pi * torch.rand(nkern)
        self.rates_ = nn.Parameter(torch.Tensor(rates))
        self.scales_ = nn.Parameter(torch.Tensor(scales))
        self.phis_ = nn.Parameter(torch.Tensor(phis))
        self.thetas_ = nn.Parameter(torch.Tensor(thetas))

    @staticmethod
    def _hs(x, scale):
        """Spectral evolution."""
        sx = scale * x
        return scale * (1-(2*math.pi*sx)**2) * torch.exp(-(2*math.pi*sx)**2/2)

    @staticmethod
    def _ht(t, rate):
        """Temporal evolution."""
        rt = rate * t
        return rate * rt**2 * torch.exp(-3.5*rt) * torch.sin(2*math.pi*rt)

    def strfs(self):
        """Make STRFs using current parameters."""
        if self.supt.device != self.rates_.device:  # for first run
            self.supt = self.supt.to(self.rates_.device)
            self.supf = self.supf.to(self.rates_.device)
        K, S, T = len(self.rates_), len(self.supf), len(self.supt)
        # Construct STRFs
        hs = self._hs(self.supf, self.scales_.view(K, 1))
        ht = self._ht(self.supt, self.rates_.view(K, 1))
        hsa = hilbert(hs, self.ndft)[:, :hs.size(-1), :]
        hta = hilbert(ht, self.ndft)[:, :ht.size(-1), :]
        hirs = hs * torch.cos(self.phis_.view(K, 1)) \
            + hsa[..., 1] * torch.sin(self.phis_.view(K, 1))
        hirt = ht * torch.cos(self.thetas_.view(K, 1)) \
            + hta[..., 1] * torch.sin(self.thetas_.view(K, 1))
        hirs_ = hilbert(hirs, self.ndft)[:, :hs.size(-1), :]  # K x S x 2
        hirt_ = hilbert(hirt, self.ndft)[:, :ht.size(-1), :]  # K x T x 2

        # for a single strf:
        # strfdn = hirt_[:, 0] * hirs_[:, 0] - hirt_[:, 1] * hirs_[:, 1]
        # strfup = hirt_[:, 0] * hirs_[:, 0] + hirt_[:, 1] * hirs_[:, 1]
        rreal = hirt_[..., 0].view(K, T, 1) * hirs_[..., 0].view(K, 1, S)
        rimag = hirt_[..., 1].view(K, T, 1) * hirs_[..., 1].view(K, 1, S)
        strfs = torch.cat((rreal-rimag, rreal+rimag), 0)  # 2K x T x S

        return strfs

    def forward(self, sigspec):
        """Convolve a spectrographic representation with all STRF kernels.

        Parameters
        ----------
        sigspec: `torch.Tensor` (batch_size, time_dim, freq_dim)
            Batch of spectrograms.
            The frequency dimension should be logarithmically spaced.

        Returns
        -------
        features: `torch.Tensor` (batch_size, nkern, time_dim, freq_dim)
            Batch of STRF activatations.

        """
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)
        strfs = self.strfs().unsqueeze(1).type_as(sigspec)
        return F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)

    def __repr__(self):
        return self.__rep


def init_STRFNet(sample_batch,
                 num_classes,
                 num_kernels=32,
                 residual_channels=[32, 32],
                 embedding_dimension=1024,
                 num_rnn_layers=2,
                 frame_rate=None, bins_per_octave=None,
                 time_support=None, frequency_support=None,
                 conv2d_sizes=(3, 3),
                 mlp_hiddims=[],
                 activate_out=nn.LogSoftmax(dim=1)
                 ):
    """Initialize a STRFNet for multi-class classification.

    This is a one-stop solution to create STRFNet and its variants.

    Parameters
    ----------
    sample_batch: [Batch,Time,Frequency] torch.FloatTensor
        A batch of training examples that is used for training.
        Some dimension parameter of the network is inferred cannot be changed.
    num_classes: int
        Number of classes for the classification task.

    Keyword Parameters
    ------------------
    num_kernels: int, 32
        2*num_kernels is the number of STRF/2D kernels.
        Doubling is due to the two orientations of the STRFs.
    residual_channels: list(int), [32, 32]
        Specify the number of conv2d channels for each residual block.
    embedding_dimension: int, 1024
        Dimension of the learned embedding (RNN output).
    frame_rate: float, None
        Sampling rate [samples/second] / hop size [samples].
        No STRF kernels by default.
    bins_per_octave: int, None
        Frequency bins per octave in CQT sense. (TODO: extend for non-CQT rep.)
        No STRF kernels by default.
    time_support: float, None
        Number of seconds spanned by each STRF kernel.
        No STRF kernels by default.
    frequency_support: int/float, None
        If frame_rate or bins_per_octave is None, interpret as GaborSTRFConv.
            - Number of frequency bins (int) spanned by each STRF kernel.
        Otherwise, interpret as STRFConv.
            - Number of octaves spanned by each STRF kernel.
        No STRF kernels by default.
    conv2d_sizes: (int, int), (3, 3)
        nn.Conv2d kernel dimensions.
    mlp_hiddims: list(int), []
        Final MLP hidden layer dimensions.
        Default has no hidden layers.
    activate_out: callable, nn.LogSoftmax(dim=1)
        Activation function at the final layer.
        Default uses LogSoftmax for multi-class classification.
    """
    if all(p is not None for p in (time_support, frequency_support)):
        is_strfnet = True
        if all(p is not None for p in (frame_rate, bins_per_octave)):
            kernel_type = 'wavelet'
        else:
            assert all(
                type(p) is int for p in (time_support, frequency_support)
            )
            kernel_type = 'gabor'
    else:
        is_strfnet = False
    is_cnn = conv2d_sizes is not None
    is_hybrid = is_strfnet and is_cnn
    if is_hybrid:
        print(f"Preparing for Hybrid STRFNet; kernel type is {kernel_type}.")
    elif is_strfnet:
        print(f"Preparing for STRFNet; kernel type is {kernel_type}.")
    elif is_cnn:
        print("Preparing for CNN.")
    else:
        raise ValueError("Insufficient parameters. Check example_STRFNet.")

    if not is_strfnet:
        strf_layer = None
    elif kernel_type == 'wavelet':
        strf_layer = STRFConv(
            frame_rate, bins_per_octave,
            time_support, frequency_support, num_kernels
        )
    else:
        strf_layer = GaborSTRFConv(
            time_support, frequency_support, num_kernels
        )

    if is_cnn:
        d1, d2 = conv2d_sizes
        if d1 % 2 == 0:
            d1 += 1
            print("Enforcing odd conv2d dimension.")
        if d2 % 2 == 0:
            d2 += 1
            print("Enforcing odd conv2d dimension.")
        conv2d_layer = nn.Conv2d(
            1, 2*num_kernels,  # Double to match the total number of STRFs
            (d1, d2), padding=(d1//2, d2//2)
        )
    else:
        conv2d_layer = None

    residual_layer = ModResnet(
        (4 if is_hybrid else 2)*num_kernels, residual_channels, False
    )
    with torch.no_grad():
        flattened_dimension = STRFNet.cnn_forward(
            sample_batch, strf_layer, conv2d_layer, residual_layer
        ).shape[-1]

    linear_layer = nn.Linear(flattened_dimension, embedding_dimension)
    rnn = nn.GRU(
        embedding_dimension, embedding_dimension, batch_first=True,
        num_layers=num_rnn_layers, bidirectional=True
    )

    mlp = MLP(
        2*embedding_dimension, num_classes, hiddims=mlp_hiddims,
        activate_hid=nn.LeakyReLU(),
        activate_out=activate_out,
        batchnorm=[True]*len(mlp_hiddims)
    )

    return STRFNet(strf_layer, conv2d_layer, residual_layer,
                   linear_layer, rnn, mlp)


class SelfAttention(nn.Module):
    """A self-attentive layer."""
    def __init__(self, indim, hiddim=256):
        super(SelfAttention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(indim, hiddim),
            nn.Tanh(),
            nn.Linear(hiddim, 1, bias=False)
        )

    def forward(self, x):
        """Transform a BxTxF input tensor."""
        y_attn = self.layers(x)
        attn = F.softmax(y_attn, dim=1)
        attn_applied = torch.matmul(x.transpose(2, 1), attn).squeeze(-1)
        return attn_applied, attn


class STRFNet(nn.Module):
    """A generic STRFNet with generic or STRF kernels in the first layer.

       Processing workflow:
       Feat. -> STRF/conv2d ->  Residual CNN -> Attention -> MLP -> Class prob.
       BxTxF ----> BxTxF -------> BxTxF ---------> BxF ----> BxK -> BxC

    """
    def __init__(self, strf_layer, conv2d_layer, residual_layer,
                 linear_layer, rnn, mlp
                 ):
        """See init_STRFNet for initializing each component."""
        super(STRFNet, self).__init__()
        self.strf_layer = strf_layer
        self.conv2d_layer = conv2d_layer
        self.residual_layer = residual_layer
        self.linear_layer = linear_layer
        self.rnn = rnn
        self.attention_layer = SelfAttention(2*rnn.hidden_size)
        self.mlp = mlp

    def forward(self, x, return_embedding=False):
        """Forward pass a batch-by-time-by-frequency tensor."""
        x = self.cnn_forward(
            x, self.strf_layer, self.conv2d_layer, self.residual_layer
        )
        x = self.linear_layer(x)
        x, _ = self.rnn(x)
        x, attn = self.attention_layer(x)
        out = self.mlp(x)

        if return_embedding:
            return out, x

        return out

    @staticmethod
    def cnn_forward(x, strf_layer, conv2d_layer, residual_layer):
        """Forward until the beginning of linear layer.

        Deals with CNN, STRFNet, or Hybrid.
        """
        def flatten(x):
            return x.transpose_(1, 2).reshape(x.size(0), x.size(1), -1)

        if strf_layer and conv2d_layer:  # Hybrid
            strf_out = strf_layer(x)
            cnn_out = conv2d_layer(x.unsqueeze(1))
            return flatten(
                residual_layer(torch.cat((strf_out, cnn_out), dim=1))
            )
        elif strf_layer:  # STRFNet
            return flatten(residual_layer(strf_layer(x)))
        else:
            return flatten(residual_layer(conv2d_layer(x.unsqueeze(1))))


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()
        self.convlayers = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        return torch.relu(residual + self.convlayers(x))


class ModResnet(nn.Module):
    """Modified ResNet from the PyTorch tutorial."""
    def __init__(self, in_chan, res_chans, pool=True):
        super(ModResnet, self).__init__()
        """Instantiate a series of residual blocks.

        Parameters
        ----------
        in_chan: int
            Input channel number
        res_chans: list(int)
            Channel number for each residual block.

        """
        self.in_channels = in_chan
        assert len(res_chans) > 0, "Requires at least one residual block!"
        res_layers = [self.make_layer(ResidualBlock, res_chans[0], 2)]
        for cc in res_chans:
            res_layers.append(self.make_layer(ResidualBlock, cc, 2, 2))

        self.res_layers = nn.Sequential(*res_layers)
        if pool:
            self.avg_pool = nn.AvgPool2d((8, 5))
        self.pool = pool

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.res_layers(x)
        if self.pool:  # average pool and then flatten out to single vector
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)

        return out


if __name__ == "__main__":
    # Test STRFNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = init_STRFNet(
        torch.rand(32, 64, 257), 2,
        time_support=10, frequency_support=2,
        #frame_rate=100, bins_per_octave=12,
        conv2d_sizes=None
    ).to(device)
    print(net)
    res = net(torch.rand(24, 64, 257).to(device))  # simulation
    loss = res.sum()
    loss.backward()
    print("Okay.")
