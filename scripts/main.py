import numpy as np
import os

import cv2
import timeit

from scripts import plotting
from scripts.a_star import AStar
from scripts.dijkstra import Dijkstra


def plot_path(path, grid):
    path_x = [path[i][0] for i in range(len(path))]
    path_y = [path[i][1] for i in range(len(path))]
    for point in zip(path_x, path_y):
        cv2.circle(grid, point, 1, 200, -1)
    grid = cv2.resize(grid, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("canvas", grid)
    if cv2.waitKey(0):
        pass


def show_im(self, im, name):
    im = cv2.resize(im, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    key = cv2.waitKey(1)
    if key == ord('s'):
        pass


def dijkstra(map, start, goal, draw=False):
    dijkstra_pp = Dijkstra(map, start, goal, "euclidean", draw=draw)
    # plot = plotting.Plotting(inflated_map, start, goal)
    start_time = timeit.default_timer()
    path, visited = dijkstra_pp.searching()
    print("Dijkstra - solution found in:", timeit.default_timer() - start_time)
    plot_path(path, grid_map)
    # plot.animation(path, visited, "Dijkstra's")


def a_star(map, start, goal, draw=False):
    astar = AStar(map, start, goal, "euclidean", draw=draw)
    # plot = plotting.Plotting(inflated_map, start, goal)
    start_time = timeit.default_timer()
    path, visited = astar.searching()
    print("A* - solution found in:", timeit.default_timer() - start_time)
    plot_path(path, grid_map)
    # plot.animation(path, visited, "A*")  # animation


if __name__ == '__main__':
    # The map resolution [m/cell]
    res = 1000000
    robot_size = 1. / res  # pixels
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    im_path = ROOT_DIR + "/../im/2.png"
    # We load map and remove uncertainty
    grid_map = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    grid_map[np.where(grid_map <= 1)] = 0
    grid_map[np.where(grid_map > 0)] =255

    # since the robot will be seen as one pixel we need to inflate all the pixels to compensate for the robot size:
    # 1. Find all solid pixels:
    indices = np.where(grid_map == 255)
    solid_pixels = zip(indices[1], indices[0])  # X,Y
    # 2. inflate using circles:
    inflated_map = grid_map.copy()
    for pixel in solid_pixels:
        inflated_map = cv2.circle(inflated_map, pixel, np.ceil(robot_size / 2).astype(int), 255, -1)
    # grid_map_resized = cv2.resize(grid_map_resized, (0, 0), fx=res, fy=res)
    scale = 1
    s_start =(290,230)
    s_goal = (34,113)

    # Check if our points are impossible:
    if inflated_map[s_start[1], s_start[0]] > 0:
        print("Start point is a collion")
    if inflated_map[s_goal[1], s_goal[0]] > 0:
        print("End point is a collion")

    tmp = inflated_map.copy()

    tmp = cv2.circle(tmp, s_start, 3, 250, -1)
    tmp = cv2.circle(tmp, s_goal, 3, 250)
    #dijkstra(inflated_map, s_start, s_goal, True)
    dijkstra(inflated_map, s_start, s_goal,True)
    a_star(inflated_map, s_start, s_goal,True)
