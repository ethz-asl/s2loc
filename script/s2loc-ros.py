import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty
from controller import Controller

DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

def fields_to_dtype(fields, point_step):
    '''
    Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

class S2LocNode(object):
    def __init__(self, pt_topic):
        rospy.init_node('S2LocNode',anonymous=True)
        self.ctrl = Controller()
        print("Accepting incoming point clouds.")
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
        dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
        cloud_arr = np.fromstring(cloud_msg.data, dtype_list)
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
        return np.reshape(cloud_arr, (cloud_msg.width * cloud_msg.height))

if __name__ == "__main__":
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode("/ply_point_cloud")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
