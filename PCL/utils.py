import numpy as np
import matplotlib.pyplot as plt


def scan2coor(scan, angle_min, increment):
    """Row lidar data to coordinates

    :param scan: input scan data
    :param angle_min:
    :param increment:
    :return: 2D array of coordinates
    """
    pcl = []
    angle = angle_min
    for s in scan:
        pcl.append(np.array([s*np.sin(angle), s*np.cos(angle)]))
        angle += increment
    return np.array(pcl)


def transform(pcl, trans_mat):
    """Transformation of coordinates"""

    n = pcl.shape[0]
    dim3 = np.ones([n, 1])
    pcl = np.concatenate([pcl, dim3], axis=1)
    # print(trans_mat.shape, pcl.shape)
    target = np.matmul(trans_mat, pcl.T).T
    target = target[:, :2]
    return np.asarray(target)


def visualize_map(map):
    plt.imshow(map * 255, cmap='gray', clim=[0, 255])
    plt.show()


def generate_target_coordinates(coordinates, x, y, angle):
    """Generate a ground truth transformation.

    :param coordinates: source coordinates
    :param x: horizontal movement
    :param y: vertical movement
    :param angle: rotation angle
    :return: target coordinates
    """
    theta = angle/180*np.pi
    transform_mat = np.mat([[np.cos(theta), np.sin(theta), x],
                            [-np.sin(theta), np.cos(theta), y],
                            [0, 0, 1]])
    target = transform(coordinates, transform_mat)
    return target
