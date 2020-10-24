import numpy as np
from PCL.utils import *

class RealtimeCorrelativeScanMatching(object):
    """
    Real-time correlative scan matching algorithm class.

    Arg:
        source: source coordinates, numpy.array
        target: target coordinates, numpy.array
        resolution: integer
        gird_size: length of each grid in high resolution lookup table, meter
        sensor_p: confidence of sensor, float, [0, 1]
        lidar_range: valid range of lidar data, meter

    Attributes:
         rasterization: Rasterize the coordinates to girds
         calc_log_likelihood: Calculate the log likelihood of the input coordinates on target map
         get_max_likelihood_in_win: Calculate the max log likelihood in given search window
         multi_resolution_csm: Implement of real-time correlative scan matching
    """

    def __init__(self, source, target, resolution, grid_size, sensor_p, lidar_range=10):
        self.source = source
        self.res = resolution
        self.grid_size = grid_size
        self.p = sensor_p
        self.lidar_range = lidar_range
        self.map, self.low_map = self.rasterization(target)

    def rasterization(self, coordinates):
        grid_coor = (coordinates / self.grid_size).astype(np.int)
        add = np.log(self.p / (1 - self.p))
        size = int(self.lidar_range/self.grid_size)
        map = np.zeros((size, size))
        for i in range(grid_coor.shape[0]):
            # lidar coordinate -> image coordinate
            x = size // 2 - grid_coor[i][1]
            y = size // 2 + grid_coor[i][0]
            if 0 <= x < size and 0 <= y < size:
                map[x][y] += add
        map = np.exp(map) / (1 + np.exp(map))
        low_map = []
        for i in range(0, map.shape[0], self.res):
            line = []
            for j in range(0, map.shape[1], self.res):
                line.append(np.max(map[i:i + self.res, j:j + self.res]))
            low_map.append(line)
        low_map = np.array(low_map)
        return map, low_map

    def calc_log_likelihood(self, approx, grid_map, grid_size):
        size = grid_map.shape[0]
        approx_coors = (approx//grid_size).astype(np.int)
        log_likelihood = 0
        for i in range(approx_coors.shape[0]):
            # lidar coordinate -> image coordinate
            x = size // 2 - approx_coors[i][1]
            y = size // 2 + approx_coors[i][0]
            if 0 <= x < size and 0 <= y < size:
                log_likelihood += grid_map[x][y]
        return log_likelihood

    def get_max_likelihood_in_win(self, coordinates, search_win):
        delta_x = search_win[0] / 2
        delta_y = search_win[1] / 2
        delta_angle = search_win[2] / 2
        max_log_likelihood = 0
        best_trans = []
        best_coors = []
        for angle in np.arange(-1 * delta_angle, delta_angle, 0.1, dtype=np.float):
            theta = angle / 180 * np.pi
            revolve_mat = np.mat([[np.cos(theta), np.sin(theta), 0],
                                  [-np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
            revolved_coors = transform(coordinates, revolve_mat)
            for x in np.arange(-delta_x, delta_x+1e-3, self.grid_size, dtype=np.float):
                for y in np.arange(-delta_y, delta_y+1e-3, self.grid_size, dtype=np.float):
                    translate_mat = np.mat([[1, 0, x],
                                            [0, 1, y],
                                            [0, 0, 1]])
                    translated_coors = transform(revolved_coors, translate_mat)
                    log_likelihood = self.calc_log_likelihood(translated_coors, self.map, self.grid_size)
                    if log_likelihood > max_log_likelihood:
                        max_log_likelihood = log_likelihood
                        best_trans = np.array([x,y,angle])
                        best_coors = translated_coors

        return max_log_likelihood, best_trans, best_coors


    def multi_resolution_csm(self, search_win):
        delta_x = search_win[0]/2
        delta_y = search_win[1]/2
        delta_angle = search_win[2]/2
        xy_step = self.grid_size*self.res
        theta_step = 1
        H_best = 0
        best_trans = []
        best_coors = []
        for angle in np.arange(-1*delta_angle, delta_angle, theta_step, dtype=np.float):
            theta = angle/180*np.pi
            revolve_mat = np.mat([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            revolved_coors = transform(self.source, revolve_mat)
            for x in np.arange(-1*delta_x, delta_x+1e-3, xy_step, dtype=np.float):
                for y in np.arange(-1*delta_y, delta_y+1e-3, xy_step, dtype=np.float):
                    translate_mat = np.mat([[1, 0, x],
                                            [0, 1, y],
                                            [0, 0, 1]])
                    translated_coors = transform(revolved_coors, translate_mat)
                    # in low resolution, when rasterizing the coordinates, the grid size should be scaled as param 'res'
                    L_i = self.calc_log_likelihood(translated_coors, self.low_map, self.res*self.grid_size)
                    if L_i < H_best:
                        # print('L_i < H_best: ', x, y, angle)
                        continue
                    H_i, H_trans, H_coors = self.get_max_likelihood_in_win(translated_coors,
                                                                           (self.res*self.grid_size,
                                                                            self.res*self.grid_size,
                                                                            theta_step*2)
                                                                           )
                    if H_i > H_best:
                        H_best = H_i
                        print(np.array([x, y, angle]), H_trans, H_best)
                        best_trans = H_trans + np.array([x, y, angle])
                        best_coors = H_coors
        return best_trans, best_coors


if __name__ == '__main__':
    laser = []
    with open('laser_data.txt', 'r') as f:
        laser = map(float, f.read().split(','))
    params = {}
    with open('param.txt', 'r') as f:
        for row in f.readlines():
            params[row.split(':')[0]] = float(row.split(':')[1])

    # row laser data to (x,y)
    coordinates = scan2coor(laser, params['angle_min'], params['angle_increment'])
    # generate target
    target = generate_target_coordinates(coordinates, x=0.25,y=-0.45,angle=3)

    realtime_csm = RealtimeCorrelativeScanMatching(coordinates, target, resolution=5, grid_size=0.02, sensor_p=0.8)

    # visualize_map(map)
    target_map,_ = realtime_csm.rasterization(target)
    visualize_map(target_map)


    best_trans, best_coors = realtime_csm.multi_resolution_csm((1, 1, 10))
    visualize_map(realtime_csm.rasterization((best_coors))[0])
    print('end: ',best_trans)
