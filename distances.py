
import numpy as np
import matplotlib.pyplot as plt
import pystan as stan

model = stan.StanModel(model_code="""
    data {
        real parallax;
        real parallax_error;
    }
    parameters {
        real<lower=0, upper=1000> d; // distance
    }
    model {
        parallax ~ normal(1.0/d, parallax_error);
    }
    """)

data = dict(parallax=0.12, parallax_error=0.06) # both in milliarcseconds/yr
p_opt = model.optimizing(data=data)
samples = model.sampling(data=data, chains=2, init=[p_opt, p_opt], iter=100000)

d_samples = samples.extract()["d"]

fig, ax = plt.subplots()
ax.hist(d_samples, bins=np.linspace(0, 100, 50))
ax.set_xlim(0, 100)
ax.set_xlabel(r"$d$ $[{\rm kpc}]$")
ax.set_yticks([])
fig.tight_layout()

fig.savefig("distances_pdf.png", dpi=150)
