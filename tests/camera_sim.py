import numpy as np


def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def extrinsic_matrix_from_rotation_and_translation(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def intrinsic_matrix_from_focal_length_and_principal_point(fx, fy, cx, cy):
    K = np.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def fov_to_focal_lenth(fov, width, deg=True):
    if deg:
        fov = np.deg2rad(fov)
    return width / (2 * np.tan(fov / 2))


def focal_lengh_to_fov(focal_length, width, deg=True):
    fov = 2 * np.arctan(width / (2 * focal_length))
    if deg:
        fov = np.rad2deg(fov)
    return fov


def get_camera_intrinsic_matrix_from_fov_and_resolution(fov, width, height, deg=True):
    fx = fov_to_focal_lenth(fov, width, deg)
    fy = fov_to_focal_lenth(fov, height, deg)
    cx = width / 2
    cy = height / 2
    return intrinsic_matrix_from_focal_length_and_principal_point(fx, fy, cx, cy)


def get_camera_intrinsic_matrix_from_focal_length_and_resolution(focal_length, width, height, deg=True):
    fx = focal_length
    fy = focal_length
    cx = width / 2
    cy = height / 2
    return intrinsic_matrix_from_focal_length_and_principal_point(fx, fy, cx, cy)


def get_camera_extrinsic_matrix_from_rotation_and_translation(roll, pitch, yaw, x, y, z):
    R = rotation_matrix_from_euler_angles(roll, pitch, yaw)
    t = np.array([x, y, z])
    return extrinsic_matrix_from_rotation_and_translation(R, t)


def project_points_to_image(points, intrinsic_matrix, extrinsic_matrix):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = np.dot(extrinsic_matrix, points.T)
    points = np.dot(intrinsic_matrix, points)
    points = points[:2, :] / points[2, :]
    return points.T


if __name__ == '__main__':
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    print(objp)
    uvw = project_points_to_image(objp, get_camera_intrinsic_matrix_from_focal_length_and_resolution(100, 640, 480), get_camera_extrinsic_matrix_from_rotation_and_translation(0, 0, 0, 0, 0, 0))
    print(uvw)