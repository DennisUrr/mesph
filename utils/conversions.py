import numpy as np

def to_cartesian_3d(rlist, philist, thetalist):
    x = rlist * np.sin(thetalist) * np.cos(philist)
    y = rlist * np.sin(thetalist) * np.sin(philist)
    z = rlist * np.cos(thetalist)
    return x, y, z

def velocities_to_cartesian_3d(vr, vphi, vtheta, rlist, philist, thetalist):
    sin_theta = np.sin(thetalist)
    cos_theta = np.cos(thetalist)
    sin_phi = np.sin(philist)
    cos_phi = np.cos(philist)

    velocities_x = vr * sin_theta * cos_phi + vtheta * cos_theta * cos_phi - vphi * sin_phi
    velocities_y = vr * sin_theta * sin_phi + vtheta * cos_theta * sin_phi + vphi * cos_phi
    velocities_z = vr * cos_theta - vtheta * sin_theta

    return velocities_x, velocities_y, velocities_z
