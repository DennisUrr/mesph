import numpy as np

def to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def spherical_to_cartesian(r, phi, theta):
    '''
    Convert spherical coordinates to cartesian coordinates.
    :param r: Radial distance
    :param phi: Azimuthal angle
    :param theta: Polar angle
    :return: x, y, z coordinates in cartesian coordinates, x, y, z coordinates are bidimensional, but has the same data in each dimension
    '''
    R, THETA = np.meshgrid(r,theta)
    PHI, R2 = np.meshgrid(phi,r)
    X = np.cos(PHI)*R2
    Y = np.sin(PHI)*R2
    Z = R*np.cos(THETA)
    #print("X.shape:", X.shape, "Y.shape:", Y.shape, "Z.shape:", Z.shape)
    return X, Y, Z

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
