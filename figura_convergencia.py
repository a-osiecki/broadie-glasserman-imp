import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

b = np.array([2, 3, 5, 10, 20, 50])
bajo = np.array([1.8865, 3.5676, 4.4204, 5.4278, 5.5712, 5.6406])
alto = np.array([6.5600, 6.8363, 5.7196, 6.1686, 6.0187, 5.8350])
verdadero = 5.731

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(b, bajo, "o-", color="tab:blue", linewidth=1.5, markersize=4, label="Bajo")
ax.plot(b, alto, "o-", color="tab:red", linewidth=1.5, markersize=4, label="Alto")
ax.axhline(verdadero, color="gray", linestyle=":", linewidth=1.5,
           label="Valor verdadero: 5.731")
ax.set_xscale("log", base=2)
ax.set_xticks(b)
ax.set_xticklabels([str(v) for v in b])
ax.minorticks_off()
ax.set_xlabel("b (escala logarítmica)")
ax.set_ylabel("Prima estimada")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("figura_convergencia.pdf")
