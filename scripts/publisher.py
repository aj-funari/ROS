import rospy
from std_msgs.msg import String
from data_recorder import data_recorder
from geometry_msgs.msg import Twist

DATA = data_recorder()
move = Twist()

def publishMethod():
    pub = rospy.Publisher('talker', Twist, queue_size=10) # definging the publisher by topic, message type
    rospy.init_node('/jackal_velocity_controller/cmd_vel', anonymous=True) # defining the ros node
    rate = rospy.Rate(10) # 10hz frequency of publisher
    
    i = 0
    while not rospy.is_shutdown():
        x_z_actions = DATA.tensor_x_z_actions
        print(x_z_actions)
        move.linear.x = 1
        move.linear.z = -1
        rospy.loginfo("Data is being sent")
        pub.publish(move)
        rate.sleep()
        i += 1

if __name__ == '__main__':
    try:
        publishMethod()
    except rospy.ROSInterruptException:
        pass