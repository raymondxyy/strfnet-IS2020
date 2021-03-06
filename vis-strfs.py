"""Visualize spectral-temporal receptive fields at different scales."""
import numpy as np
import scipy.signal as signal


class HilbertTransformer(object):
    def __init__(self, method='frequency_sampling', ndft=None):
        if method == 'frequency_sampling':
            def hilbert(sig):
                return signal.hilbert(sig, ndft)[:len(sig)]
            self._fn = hilbert
        elif method == 'pm':
            ht = signal.remez(
                201, [0.01, 0.99], [1], type='hilbert', fs=2
            )
            def hilbert(sig):
                sig_imag = np.convolve(sig, ht)[100:-100]
                return sig + 1j*sig_imag
            self._fn = hilbert
        else:
            raise NotImplementedError

    def __call__(self, sig):
        return self._fn(sig)


def strf(time, freq, sr, bins_per_octave, rate=1, scale=1, phi=0, theta=0,
         ndft=None):
    """Spectral-temporal response fields for both up and down direction.

    Implement the STRF described in Chi, Ru, and Shamma:
    Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal
    analysis of complex sounds. The Journal of the Acoustical Society of
    America, 118(2), 887–906. https://doi.org/10.1121/1.1945807.

    Parameters
    ----------
    time: int or float
        Time support in seconds. The returned STRF will cover the range
        [0, time).
    freq: int or float
        Frequency support in number of octaves. The returned STRF will
        cover the range [-freq, freq).
    sr: int
        Sampling rate in Hz.
    bins_per_octave: int
        Number of frequency bins per octave on the log-frequency scale.
    rate: int or float
        Stretch factor in time.
    scale: int or float
        Stretch factor in frequency.
    phi: float
        Orientation of spectral evolution in radians.
    theta: float
        Orientation of time evolution in radians.

    """
    def _hs(x, scale):
        """Construct a 1-D spectral impulse response with a 2-diff Gaussian.

        This is the prototype filter suggested by Chi et al.
        """
        sx = scale * x
        return scale * (1-(2*np.pi*sx)**2) * np.exp(-(2*np.pi*sx)**2/2)

    def _ht(t, rate):
        """Construct a 1-D temporal impulse response with a Gamma function.

        This is the prototype filter suggested by Chi et al.
        """
        rt = rate * t
        return rate * rt**2 * np.exp(-3.5*rt) * np.sin(2*np.pi*rt)

    hs = _hs(np.linspace(-freq, freq, endpoint=False,
             num=int(2*freq*bins_per_octave)), scale)
    ht = _ht(np.linspace(0, time, endpoint=False, num=int(sr*time)), rate)
    if ndft is None:
        ndft = max(512, nextpow2(max(len(hs), len(ht))))
        ndft = max(len(hs), len(ht))
    assert ndft >= max(len(ht), len(hs))
    hilb = HilbertTransformer(method='pm')
    hsa = hilb(hs)
    hta = hilb(ht)
    #hsa = signal.hilbert(hs, ndft)[:len(hs)]
    #hta = signal.hilbert(ht, ndft)[:len(ht)]
    hirs = hs * np.cos(phi) + hsa.imag * np.sin(phi)
    hirt = ht * np.cos(theta) + hta.imag * np.sin(theta)
    hirs_ = hilb(hirs)
    hirt_ = hilb(hirt)
    #hirs_ = signal.hilbert(hirs, ndft)[:len(hs)]
    #hirt_ = signal.hilbert(hirt, ndft)[:len(ht)]
    return np.outer(hirt_, hirs_).real,\
        np.outer(np.conj(hirt_), hirs_).real


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    SR = 16000
    CQT_FRATE = 100
    BPO = 6
    TSUPP = float(input("Enter the time support in seconds [0.5]: ") or "0.5")
    FSUPP = float(input("Enter the frequency support in octaves [2.0]: ") or "2")
    print(f"""Visualization of a STRF pair as a function of rate and scale.
       Configs: {SR} Hz sampling rate
             {CQT_FRATE} Hz frame rate
             {BPO} bins per octave
             {TSUPP} seconds time support
             {FSUPP} octaves frequency support""")


    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for ss in range(1, 26):
        for rr in range(1, 26):
            phi = 0
            theta = 0
            scale = ss / 5
            rate = rr / 2
            kdn, kup = strf(TSUPP, FSUPP, CQT_FRATE, BPO, rate=rate, scale=scale,
                            phi=phi*np.pi, theta=theta*np.pi, ndft=512)

            for xx, kk in zip(ax, (kdn, kup)):
                xx.clear()
                dbspec = 10*np.log10(np.clip(kk**2, 1e-8, None))
                xx.pcolormesh(
                    *np.mgrid[slice(0, TSUPP, TSUPP/len(kk)),
                              slice(-FSUPP, FSUPP, 2*FSUPP/int(2*FSUPP*BPO))],
                    dbspec,
                    cmap='jet',
                    vmin=-60,
                    vmax=0,
                    shading='auto'
                )
                fig.canvas.draw()
                fig.suptitle(
                    f'''Support: {TSUPP*1e3:.1f} ms, {FSUPP:.1f} octaves
                    Rate: {rate:.1f} Hz, Scale: {scale:.1f} cycles/octave''',
                    fontsize=16)
                plt.pause(.5)

    plt.close(fig)
