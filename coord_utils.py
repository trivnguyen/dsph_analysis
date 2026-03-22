"""Coordinate transformation utilities."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import astropy.units as u


def rotation_matrix_from_vectors(
    vec1: NDArray[np.floating],
    vec2: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Find the rotation matrix that aligns vec1 to vec2.

    Parameters
    ----------
    vec1 : NDArray[np.floating]
        Source vector (3D).
    vec2 : NDArray[np.floating]
        Destination vector (3D).

    Returns
    -------
    NDArray[np.floating]
        3x3 rotation matrix that transforms vec1 to align with vec2.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def cartesian_to_spherical(
    pos: u.Quantity,
    vel: u.Quantity
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Convert Cartesian coordinates and velocities to spherical.

    Adapted from EinsteinPy:
    https://github.com/einsteinpy/einsteinpy/blob/main/src/einsteinpy/coordinates/utils.py

    Parameters
    ----------
    pos : u.Quantity
        Position array of shape (N, 3) with units.
    vel : u.Quantity
        Velocity array of shape (N, 3) with units.

    Returns
    -------
    r : u.Quantity
        Radial distance.
    theta : u.Quantity
        Polar angle (from z-axis).
    phi : u.Quantity
        Azimuthal angle (in x-y plane from x-axis).
    v_r : u.Quantity
        Radial velocity.
    v_theta : u.Quantity
        Polar velocity component.
    v_phi : u.Quantity
        Azimuthal velocity component.
    """
    x, y, z = pos.T
    v_x, v_y, v_z = vel.T

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    v_r = (x * v_x + y * v_y + z * v_z) / r
    v_phi = (v_y * x - v_x * y) / np.sqrt(x**2 + y**2)
    v_theta = (v_r * z - v_z * r) / np.sqrt(x**2 + y**2)

    return r, theta, phi, v_r, v_theta, v_phi
