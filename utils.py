"""Utility functions for STRFNet."""
import torch
import torch.nn as nn

def nextpow2(n):
    """Give next power of 2 bigger than n."""
    return 1 << (n-1).bit_length()

def hilbert(x, ndft=None):
    r"""Analytic signal of x.

    Return the analytic signal of a real signal x, x + j\hat{x}, where \hat{x}
    is the Hilbert transform of x.

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

    return torch.ifft(xspec * h, 1)


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
