
import numpy as np
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import gala.dynamics as gd
import gala.potential as gp
from astropy import units as u

np.random.seed(42) 
# Note this is not totally reproducible because we may get different sources
# back from Gaia


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




# Retrieve a single Gaia source.
'''
job = Gaia.launch_job("""
    SELECT TOP 1 * 
    FROM    gaiadr2.gaia_source
    WHERE   parallax > 0
    AND     parallax_over_error > 8
    AND     abs(pmra/pmra_error) > 5
    AND     abs(pmdec/pmdec_error) > 5
    AND     abs(radial_velocity) < 100
    """)
'''
# For reproducibility:
job = Gaia.launch_job("""
    SELECT  *
    FROM    gaiadr2.gaia_source
    WHERE   source_id = 4099457674642645248
    """)
gaia_source = job.get_results()[0]

# Set up our Milky Way potential.
n_steps = 1000
dt = -0.5 * u.Myr
potential = gp.MilkyWayPotential()
v_sun = coord.CartesianDifferential([11.1, 250, 7.25]*u.km/u.s)
gc_frame = coord.Galactocentric(galcen_distance=8.3*u.kpc,
                                z_sun=0*u.pc,
                                galcen_v_sun=v_sun)


mu, cov = astrometry_covariance_matrix(gaia_source)


# Make 100 draws of the initial position.
N = 100
astrometric_draws = np.random.multivariate_normal(mu, cov, N)
radial_velocity_draws = np.random.normal(
    gaia_source["radial_velocity"], gaia_source["radial_velocity_error"], N)


orbits = []
for i in range(N):

    icrs = coord.ICRS(
        ra=astrometric_draws[i, 0] * u.deg,
        dec=astrometric_draws[i, 1] * u.deg,
        distance=1.0/astrometric_draws[i, 2] * u.kpc,
        pm_ra_cosdec=astrometric_draws[i, 3] * u.mas/u.yr,
        pm_dec=astrometric_draws[i, 4] * u.mas/u.yr,
        radial_velocity=radial_velocity_draws[i] * u.km/u.s)

    gc  = icrs.transform_to(gc_frame)
    w0 = gd.PhaseSpacePosition(gc.data)
    orbits.append(potential.integrate_orbit(w0, dt=dt, n_steps=n_steps))

# Just do the orbit as given by Gaia.
icrs = coord.ICRS(
    ra=gaia_source["ra"] * u.deg,
    dec=gaia_source["dec"] * u.deg,
    distance=1.0 / gaia_source["parallax"] * u.kpc,
    pm_ra_cosdec=gaia_source["pmra"] * u.mas/u.yr,
    pm_dec=gaia_source["pmdec"] * u.mas/u.yr,
    radial_velocity=gaia_source["radial_velocity"] * u.km/u.s)

gc = icrs.transform_to(gc_frame)
w0 = gd.PhaseSpacePosition(gc.data)
orbit = potential.integrate_orbit(w0, dt=dt, n_steps=n_steps)

kwds = dict(c="#666666", alpha=0.1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(orbit.x, orbit.y, c="k", lw=2, zorder=10)
axes[1].plot(orbit.x, orbit.z, c="k", lw=2, zorder=10)
axes[2].plot(orbit.y, orbit.z, c="k", lw=2, zorder=10)

for i, orbit in enumerate(orbits):

    axes[0].plot(orbit.x, orbit.y, **kwds)
    axes[1].plot(orbit.x, orbit.z, **kwds)
    axes[2].plot(orbit.y, orbit.z, **kwds)


axes[0].set_xlabel(r"$x$ $[{\rm kpc}]$")
axes[0].set_ylabel(r"$y$ $[{\rm kpc}]$")
axes[1].set_xlabel(r"$x$ $[{\rm kpc}]$")
axes[1].set_ylabel(r"$z$ $[{\rm kpc}]$")
axes[2].set_xlabel(r"$y$ $[{\rm kpc}]$")
axes[2].set_ylabel(r"$x$ $[{\rm kpc}]$")

fig.tight_layout()

fig.savefig("orbits.png", dpi=150)
