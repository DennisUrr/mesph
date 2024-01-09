import numpy as np

def to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # Asegúrate de que r no sea 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def to_spherical_velocity(x, y, z, vx, vy, vz):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)

    # Asegurarse de que no hay divisiones por cero
    if r == 0 or r_xy == 0:
        raise ValueError("r o r_xy es 0, lo que podría llevar a una división por cero.")

    vr = (x * vx + y * vy + z * vz) / r
    vtheta = (x * z * vx + y * z * vy - (x**2 + y**2) * vz) / (r * r_xy)
    vphi = (-y * vx + x * vy) / r_xy

    return vr, vtheta, vphi

def velocities_to_cartesian_3d(vr, vphi, vtheta, rlist, philist, thetalist):
    """
    Converts velocity components from spherical to Cartesian coordinates in 3D.

    :param vr: Array of radial velocity components in spherical coordinates.
    :param vphi: Array of azimuthal velocity components in spherical coordinates.
    :param vtheta: Array of polar velocity components in spherical coordinates.
    :param rlist, philist, thetalist: Arrays of spherical coordinates (r, phi, theta).
    :return: Three arrays representing the velocity components in x, y, and z directions in Cartesian coordinates.

    This function transforms the velocity components in spherical coordinates to their
    corresponding components in Cartesian coordinates, taking into account the positions
    of particles in both coordinate systems.
    """
    sin_theta = np.sin(thetalist)
    cos_theta = np.cos(thetalist)
    sin_phi = np.sin(philist)
    cos_phi = np.cos(philist)

    velocities_x = vr * sin_theta * cos_phi + vtheta * cos_theta * cos_phi - vphi * sin_phi
    velocities_y = vr * sin_theta * sin_phi + vtheta * cos_theta * sin_phi + vphi * cos_phi
    velocities_z = vr * cos_theta - vtheta * sin_theta
    # print("Velocity sampling completed.")

    return velocities_x, velocities_y, velocities_z
