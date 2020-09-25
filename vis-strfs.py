"""Visualize spectral-temporal receptive fields at different scales."""
import matplotlib.pyplot as plt
import numpy as np
from audlib.sig.spectemp import strf


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
