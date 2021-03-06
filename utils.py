"""Utility functions for STRFNet."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import scipy.signal as signal


def design_hilbert_transformer():
    ht = signal.remez(301, [0.01, 0.99], [1], type='hilbert', fs=2)
    return ht

def nextpow2(n):
    """Give next power of 2 bigger than n."""
    return 1 << (n-1).bit_length()

def _hilbert_legacy(x, ndft=None):
    r"""Return the Hilbert tranform of x.

    Parameters
    ----------
    x: torch.Tensor
        Audio signal to be analyzed.
        Always assumes x is real, and x.shape[-1] is the signal length.

    Returns
    -------
    out: torch.Tensor
        out.shape == (*x.shape, 2)

    """
    if ndft is None:
        sig = x
    else:
        assert ndft > x.size(-1)
        sig = F.pad(x, (0, ndft-x.size(-1)))
    xspec = torch.rfft(sig, 1, onesided=False)
    siglen = sig.size(-1)
    h = torch.zeros(siglen, 2, dtype=sig.dtype, device=sig.device)
    if siglen % 2 == 0:
        h[0] = h[siglen//2] = 1
        h[1:siglen//2] = 2
    else:
        h[0] = 1
        h[1:(siglen+1)//2] = 2

    return torch.ifft(xspec * h, 1)[..., :x.size(-1), 1]

def hilbert(x, n=None):
    """Hilbert transform using PyTorch.fft."""
    xspec = fft.fft(x, n)
    fresp = torch.zeros_like(xspec[0])

    freq_dim = xspec.shape[-1]
    if freq_dim % 2:
        fresp[0] = 1
        fresp[1:(freq_dim + 1) // 2] = 2
    else:
        fresp[0] = fresp[freq_dim//2] = 1
        fresp[1:(freq_dim//2)] = 2

    out = fft.ifft(xspec * fresp)
    return out[..., :x.size(-1)].imag


class HilbertTranformer(nn.Module):
    def __init__(self, method='frequency-sampling', ndft=None):
        super().__init__()
        self.method = method
        if method == 'frequency-sampling':
            self.ndft = ndft
        elif self.method == 'pm':
            # NOTE: pre-flip so torch can do Conv1d
            ht = design_hilbert_transformer().tolist()[::-1]
            assert len(ht) % 2, "Design Type III filter only!"
            self.ht = torch.FloatTensor(ht)[None, None, ...]
        else:
            raise NotImplementedError

    def forward(self, x):
        """Return the analytic signal of x."""
        if self.method == 'frequency-sampling':
            return hilbert(x, self.ndft)
        elif self.method == 'pm':
            if self.ht.device != x.device:
                self.ht = self.ht.to(x.device)
            x_imag = F.conv1d(
                x.unsqueeze(1), self.ht, padding=self.ht.size(-1)//2
            )
            return x_imag.squeeze(1)
        else:
            raise NotImplementedError


class MLP(nn.Module):
    """Multi-Layer Perceptron."""

    def __init__(self, indim, outdim, hiddims=[], bias=True,
                 activate_hid=nn.ReLU(), activate_out=nn.ReLU(),
                 batchnorm=[]):
        """Initialize a MLP.

        Parameters
        ----------
        indim: int
            Input dimension to the MLP.
        outdim: int
            Output dimension to the MLP.
        hiddims: list of int
            A list of hidden dimensions. Default ([]) means no hidden layers.
        bias: bool [True]
            Apply bias for this network?
        activate_hid: callable, optional
            Activation function for hidden layers. Default to ReLU.
        activate_out: callable, optional
            Activation function for output layer. Default to ReLU.

        """
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hiddims = hiddims
        self.nhidden = len(hiddims)
        if self.nhidden == 0:
            print("No hidden layers.")
        indims = [indim] + hiddims
        outdims = hiddims + [outdim]
        self.layers = nn.ModuleList([])
        for ii in range(self.nhidden):
            self.layers.append(nn.Linear(indims[ii], outdims[ii], bias=bias))
            if len(batchnorm) > 0 and batchnorm[ii]:
                self.layers.append(nn.BatchNorm1d(outdims[ii], momentum=0.05))
            self.layers.append(activate_hid)
        self.layers.append(nn.Linear(indims[-1], outdims[-1], bias=bias))
        if activate_out is not None:
            self.layers.append(activate_out)

    def forward(self, x):
        """One forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from audlib.quickstart import welcome

    # Choose input
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    x = signal.chirp(t, 20., t[-1], 100.)
    x *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) * np.exp(-5*t))
    x = torch.from_numpy(x)[None, ...].float()
    #x, _ = welcome()
    #x = torch.from_numpy(x[1000:1320])[None, ...].float()

    # Processing starts here
    hx1 = _hilbert_legacy(x, 512)
    hx2 = hilbert(x, 512)
    assert torch.allclose(hx1, hx2)

    hb = HilbertTranformer('pm')
    hx3 = hb(x)

    fig, ax = plt.subplots()
    ax.plot(x[0].numpy(), label='Signal')
    ax.plot(
        (hx1[0].numpy()**2 + x[0].numpy()**2)**.5,
        label='Legacy'
    )
    ax.plot(
        (hx2[0].numpy()**2 + x[0].numpy()**2)**.5,
        label='Frequency sampling'
    )
    ax.plot(
        (hx3[0].numpy()**2 + x[0].numpy()**2)**.5,
        label='Parks-McClellan'
    )
    ax.legend()
    plt.show()
