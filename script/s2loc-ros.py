import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty
from controller import Controller

class S2LocNode(object):
    def __init__(self, pt_topic):
        rospy.init_node('S2LocNode',anonymous=True)
        self.ctrl = Controller()
        print("Waiting for point clouds.")
        self.is_detecting = False
        self.pc_sub = rospy.Subscriber(pt_topic, PointCloud2, self.laser_callback)
        self.lc_service = rospy.Service('s2loc_detect', Empty, self.detect_lc)

    def laser_callback(self, cloud_msg):
        if self.is_detecting:
            return
        cloud = self.__convert_msg_to_array(cloud_msg)
        print(f'Received pc with size {cloud.size}  and shape {cloud.shape}')
        self.ctrl.handle_point_cloud(cloud_msg.header.stamp, cloud)

    def detect_lc(self, request):
        self.is_detecting = True
        self.ctrl.find_loop_closures()
        self.ctrl.clear_clouds()
        self.is_detecting = False

    def __convert_msg_to_array(self, cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

if __name__ == "__main__":
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode("/point_cloud")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
