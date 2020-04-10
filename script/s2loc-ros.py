import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class S2LocNode(object):
    def __init__(self, pt_topic):
        rospy.init_node('S2LocNode',anonymous=True)
        rospy.Subscriber(pt_topic, PointCloud2, self.laser_callback)


    def laser_callback(self, cloud):
        print("received pc")

if __name__ == "__main__":
    try:
        s2loc = S2LocNode("/ply_point_cloud")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
