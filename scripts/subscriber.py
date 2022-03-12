import rospy
from std_msgs.msg import String
from data_recorder import data_recorder

i = 0
def publishMethod():
    pub = rospy.Publisher('talker', String, queue_size=10) # definging the publisher by topic, message type
    ### What node am I pushing to? cmd_vel?
    rospy.init_node('/jackal_velocity_controller/cmd_vel', anonymous=True) # defining the ros node - publish node
    rate = rospy.Rate(10) # 10hz frequency of publisher
    while not rospy.is_shutdown():
        data = data_recorder.training_data
        rospy.loginfo("Data is being sent")
        pub.publish(data)
        rate.sleep()
        i += 1

if __name__ == '__main__':
    try:
        publishMethod()
    except rospy.ROSInterruptException:
        pass