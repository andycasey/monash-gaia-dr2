

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

N = 40

x = np.random.uniform(2015.5, 2016.5, size=N)


m_true = 1.3

yerr = 0.1 * np.random.normal(0, 1, size=N)
y_high_snr = m_true * (x - x.mean()) + yerr

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].axhline(0, c="#666666", linestyle=":", zorder=-100)
axes[0].scatter(x, y_high_snr, c="k", zorder=10)
axes[0].errorbar(x, y_high_snr, yerr=yerr, fmt=None, ecolor="#666666", zorder=-1)

axes[0].set_xlabel("epoch")
axes[0].set_ylabel(r"$y - \mathcal{M}(\alpha,\delta,\mu_\alpha^*,\mu_\delta)$")

xlims = np.array([x.min(), x.max()])
axes[0].plot(xlims, m_true * (xlims - np.mean(xlims)), c='r', lw=2)
axes[0].set_xlim(*xlims)


A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y_high_snr)))

axes[1].hist(np.random.normal(m_ls, cov[1, 1]**0.5, size=10000), bins=50,
    facecolor="r")
axes[1].set_xlabel(r"$m$")

fig.tight_layout()
fig.savefig("negative_parallaxes_high_snr.png", dpi=150)

m_true = 1e-3
yerr = 1.0 * np.random.normal(0, 1, size=N)
y_low_snr = m_true * (x - x.mean()) + yerr

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].axhline(0, c="#666666", linestyle=":", zorder=-100)
axes[0].scatter(x, y_low_snr, c="k", zorder=10)
axes[0].errorbar(x, y_low_snr, yerr=yerr, fmt=None, ecolor="#666666", zorder=-1)

axes[0].set_xlabel("epoch")
axes[0].set_ylabel(r"$y - \mathcal{M}(\alpha,\delta,\mu_\alpha^*,\mu_\delta)$")

xlims = np.array([x.min(), x.max()])
axes[0].plot(xlims, m_true * (xlims - np.mean(xlims)), c='r', lw=2)
axes[0].set_xlim(*xlims)

A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y_low_snr)))

axes[1].hist(np.random.normal(m_ls, cov[1, 1]**0.5, size=10000), bins=50,
    facecolor="r")
axes[1].set_xlabel(r"$m$")

fig.tight_layout()
fig.savefig("negative_parallaxes_low_snr.png", dpi=150)
