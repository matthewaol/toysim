import numpy as np


def qxyz_from_det(detector, beam):
    """

    Parameters
    ----------
    detector: dxtbx detector model
    beam: dxtbx beam model

    Returns
    -------

    """
    wavelen = beam.get_wavelength()
    unit_s0 = beam.get_unit_s0()

    xdim, ydim = detector[0].get_image_size()

    Npix_per_pan = xdim * ydim

    Qvecs = np.zeros((len(detector), Npix_per_pan, 3))

    for pid in range(len(detector)):
        FAST = np.array(detector[pid].get_fast_axis())
        SLOW = np.array(detector[pid].get_slow_axis())
        ORIG = np.array(detector[pid].get_origin())

        Ypos, Xpos = np.indices((ydim, xdim))
        px = detector[pid].get_pixel_size()[0]
        Ypos = Ypos * px
        Xpos = Xpos * px

        SX = ORIG[0] + FAST[0] * Xpos + SLOW[0] * Ypos
        SY = ORIG[1] + FAST[1] * Xpos + SLOW[1] * Ypos
        SZ = ORIG[2] + FAST[2] * Xpos + SLOW[2] * Ypos

        Snorm = np.sqrt(SX ** 2 + SY ** 2 + SZ ** 2)

        SX /= Snorm
        SY /= Snorm
        SZ /= Snorm

        QX = (SX - unit_s0[0]) /wavelen
        QY = (SY - unit_s0[1]) /wavelen
        QZ = (SZ - unit_s0[2]) /wavelen

        Qvecs[pid, :,0]= QX.ravel()
        Qvecs[pid, :,1]= QY.ravel()
        Qvecs[pid, :,2]= QZ.ravel()

    return Qvecs

