#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def publishMethod():
    pub = rospy.Publisher('talker', String, queue_size=10) # definging the publisher by topic, message type
    rospy.init_node('publish_node', anonymous=True) # defining the ros node - publish node
    rate = rospy.Rate(10) # 10hz frequency of publisher
    while not rospy.is_shutdown():
        publishString = "This is being published"
        rospy.loginfo("Data is being sent")
        pub.publish(publishString)
        rate.sleep()

if __name__ == '__main__':
    try:
        publishMethod()
    except rospy.ROSInterruptException:
        pass
