import numpy as np

def to_cartesian_3d(rlist, philist, thetalist):
    """
    Converts spherical coordinates to Cartesian coordinates in 3D.

    :param rlist: Array of radial distances (r component in spherical coordinates).
    :param philist: Array of azimuthal angles (phi component in spherical coordinates).
    :param thetalist: Array of polar angles (theta component in spherical coordinates).
    :return: Three arrays representing the x, y, and z coordinates in Cartesian coordinate system.

    This function takes arrays of radial distances, azimuthal angles, and polar angles in
    spherical coordinates and converts them to the x, y, and z coordinates in the Cartesian
    coordinate system.
    """
    x = rlist * np.sin(thetalist) * np.cos(philist)
    y = rlist * np.sin(thetalist) * np.sin(philist)
    z = rlist * np.cos(thetalist)
    print("Particle sampling completed.")
    return x, y, z

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
    print("Velocity sampling completed.")

    return velocities_x, velocities_y, velocities_z
