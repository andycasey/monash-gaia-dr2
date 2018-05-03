
from astroquery.gaia import Gaia

job = Gaia.launch_job("""
    SELECT parallax, parallax_error
    FROM gaiadr2.gaia_source
    WHERE MOD(random_index, 1000000) = 0
    """)


