#!/usr/bin/env python
import copy
import numpy as np
import os

import rospy
import tf2_ros
from nav_msgs.msg import OccupancyGrid, MapMetaData
import cv2

from visualization_msgs.msg import Marker


def occupancygrid_to_numpy(msg):
    data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    return np.ma.array(data, mask=data == -1, fill_value=-1)


def numpy_to_occupancy_grid(arr, info=None):
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')
    if not arr.dtype == np.int8:
        raise TypeError('Array must be of int8s')

    grid = OccupancyGrid()
    if isinstance(arr, np.ma.MaskedArray):
        # We assume that the masked value are already -1, for speed
        arr = arr.data
    grid.data = arr.ravel()
    grid.info = info or MapMetaData()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]

    return grid


# https://github.com/jhan15/exploration_robot
class MapNode:
    def __init__(self):

        # Creates a node with subscriber
        rospy.init_node('a_star')
        self.timer = rospy.Timer(rospy.Duration(0, 5e+6), self.map_visualizer_callback)  # this is a timer to output
        self.marker_pub = rospy.Publisher("/test_marker", Marker, queue_size=1)
        # self.global_costmap_sub = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid,
        #                                            self.costmap_callback)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        self.robot_frame_id = rospy.get_param("~robot_frame_id", "base_link")
        # Create TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.grid_map = None
        self.map = None
        self.res = None
        self.bbx_min = []
        self.bbx_max = []
        self.save_im = False
        while self.map is None:
            if rospy.is_shutdown():
                rospy.loginfo("shutting down")
                return
            rospy.loginfo("Explorer: No map recieved yet!")
            rospy.sleep(1)
        print("hey")

    def costmap_callback(self, msg):
        self.grid_map_msg = msg
        self.map = msg
        self.unpack_map_msg(msg)

    def unpack_map_msg(self, msg):
        map = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        # Save maps and map info
        self.grid_map = map
        self.res = msg.info.resolution
        # Determine the area of interest
        indices = np.where(map > 0)
        self.bbx_min = [indices[0].min(), indices[1].min()]  # Height, width
        self.bbx_max = [indices[0].max(), indices[1].max()]  # Height, width
        print("map updated")

    def map_callback(self, msg):
        self.map = msg
        self.unpack_map_msg(msg)

    def goal_point_callback(self, msg):
        pass

    def make_marker(self, robot_position):
        marker_ = Marker()
        marker_.header.frame_id = "map"
        # marker_.header.stamp = rospy.Time.now()
        marker_.type = marker_.CUBE
        marker_.action = marker_.ADD

        marker_.pose.position.x = robot_position[0][0]
        marker_.pose.position.y = robot_position[0][1]
        marker_.pose.position.z = 0
        marker_.pose.orientation.x = 0
        marker_.pose.orientation.y = 0
        marker_.pose.orientation.z = 0
        marker_.pose.orientation.w = 1

        marker_.lifetime = rospy.Duration.from_sec(10)
        marker_.scale.x = 0.5
        marker_.scale.y = 0.5
        marker_.scale.z = 0.5
        marker_.color.a = 0.5
        marker_.color.r = 0
        marker_.color.g = 0
        marker_.color.b = 255
        return marker_

    def map_visualizer_callback(self, arg):
        if self.res is None:
            return
        if self.grid_map is None:
            return
        if len(self.bbx_min) == 0 or len(self.bbx_max) == 0:
            return

        self.tf_buffer.can_transform("map", self.robot_frame_id, rospy.Time(0), rospy.Duration(10))
        h, w = self.grid_map.shape[:2]
        # Get root position based on TF
        robot_pose = self.tf_buffer.lookup_transform("map", self.robot_frame_id, rospy.Time(0))
        robot_position = np.array(
            [robot_pose.transform.translation.x, robot_pose.transform.translation.y])
        robot_map_position = ((w / 2 + (robot_position / self.res))).astype(int)
        tmp = copy.deepcopy(self.grid_map)
        cropped = tmp[self.bbx_min[0]:self.bbx_max[0], self.bbx_min[1]: self.bbx_max[1]]
        cropped += abs(cropped.min())
        robot_cropped_position = robot_map_position[0] - self.bbx_min[1], robot_map_position[1] - self.bbx_min[0]
        color_im = self.convert_map(cropped, robot_cropped_position)
        if self.save_im:
            s_im = color_im = self.convert_map(cropped)
            s_im = np.fliplr(s_im)
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            cv2.imwrite(ROOT_DIR + "/../im/map.png", s_im)
            self.save_im = False
            print("im saved", ROOT_DIR + "/../im/map.png")

        color_im = np.fliplr(color_im)
        self.show_im(color_im, "map")

    def convert_map(self, map, robot_position=None):
        map = map.astype(np.uint8)
        map_color = cv2.cvtColor(map, cv2.COLOR_GRAY2RGB)
        if robot_position is not None:
            map_color = cv2.circle(map_color, robot_position, 3, (0, 0, 255), -1)  # cv2 takes (x,y)
        return map_color

    def show_im(self, im, name):
        cv2.imshow(name, im)
        key = cv2.waitKey(1)
        if key == ord('s'):
            self.save_im = True


if __name__ == '__main__':
    try:
        x = MapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print ("error!")
