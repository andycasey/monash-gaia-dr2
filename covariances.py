
import numpy as np
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt

def astrometry_covariance_matrix(source):
    """
    Construct a covariance matrix for a 5-parameter astrometric solution 
    measured by Gaia for a single source.

    :param source:
        A dictionary-like object (or table row) containing the requisite 
        information about the source: `ra`, `dec`, `parallax`, `pmra`, `pmdec`
        as well as associated errors (`*_error`) and correlation coefficients
        (e.g., `ra_dec_corr`).

    :returns:
        The multidimensional mean and covariance matrix.
    """

    terms = ("ra", "dec", "parallax", "pmra", "pmdec")
    mu = np.array([source[term] for term in terms])
    cov_diag = np.atleast_2d([source["{}_error".format(term)] for term in terms])

    cov = np.dot(cov_diag.T, cov_diag)

    for i, i_term in enumerate(terms):
        for j, j_term in enumerate(terms):
            if i >= j: continue
        
            term_name = "{}_{}_corr".format(i_term, j_term)
            cov[i, j] *= source[term_name]
            cov[j, i] *= source[term_name]

    return (mu, cov)


def astrometry_correlation_matrix(source):
    """
    Construct a correlation matrix for a 5-parameter astrometric solution 
    measured by Gaia for a single source.

    :param source:
        A dictionary-like object (or table row) containing the requisite 
        information about the source: `ra`, `dec`, `parallax`, `pmra`, `pmdec`
        as well as associated errors (`*_error`) and correlation coefficients
        (e.g., `ra_dec_corr`).

    :returns:
        The correlation matrix between parameters of the astrometric solution.
    """

    terms = ("ra", "dec", "parallax", "pmra", "pmdec")
    T = len(terms)
    rho = np.ones((T, T))
    for i, i_term in enumerate(terms):
        for j, j_term in enumerate(terms):
            if i >= j: continue

            term_name = "{}_{}_corr".format(i_term, j_term)
            rho[i, j] = source[term_name]
            rho[j, i] = source[term_name]

    return rho




# Retrieve a single Gaia source.
job = Gaia.launch_job("""
    SELECT TOP 1 * 
    FROM    gaiadr2.gaia_source
    WHERE   parallax IS NOT NULL
    AND     pmra IS NOT NULL
    AND     pmdec IS NOT NULL
    """)
gaia_source = job.get_results()[0]

mu, cov = astrometry_covariance_matrix(gaia_source)

# Make 1,000 draws of the initial position.
draws = np.random.multivariate_normal(mu, cov, 1000)

# Inspect the correlation matrix (for fun)
rho = astrometry_correlation_matrix(gaia_source)

fig, ax = plt.subplots()
image = ax.imshow(rho)
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
labels = (r"$\alpha$", r"$\delta$", r"$\varpi$", r"$\mu_{\alpha^*}$",
    r"$\mu_{\delta}$")
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

cbar = plt.colorbar(image)

fig.tight_layout()
fig.savefig("covariances.png", dpi=150)
