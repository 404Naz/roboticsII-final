import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math
import time
from collections import deque

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        self.start_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)
        self.sub_detected_start_pose = self.create_subscription(PoseStamped, 'detected_color_start_pose', self.detected_start_pose_callback, 10)
        self.sub_start_tracking = self.create_subscription(String, '/start_tracking', self.start_tracking_callback, 10)

        self.start_position = np.zeros(2)
        self.approach = True
        self.set_start_position = True

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.05, self.timer_update)
        self.counter = 0
        self.counter2 = 0
        self.step_count = 0
        self.steps = deque(maxlen=500)
        self.steps.append(Twist())
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # self.get_logger().info("Recieved Object Pose")
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 1: # or center_points[2] > 0.7:
            # self.obs_pose = None
            # return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_start_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if not (np.linalg.norm(center_points) > 0.1 and center_points[2] < 0.1):
        #     self.start_pose = None
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.start_pose = cp_world
        # self.get_logger().info(f"Recieved Start Pose, start: {self.start_pose}, goal: {self.goal_pose}")

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # self.get_logger().info(f"Recieved Goal Pose, {np.linalg.norm(center_points)}")
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        if np.linalg.norm(center_points) < 0.3 or not self.approach: #or center_points[2] > 0.7:
            self.approach = False
            self.goal_pose = self.start_position
            return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.goal_pose = cp_world

    def start_tracking_callback(self, msg):
        self.get_logger().info("In start callback")
        if msg.data == "Start":
            self.set_start_position = False
            self.approach = True
            self.steps = deque(maxlen=500)
            self.step_count = 0
        else:
            self.set_start_position = True
            self.approach = True

    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            robot_pos = np.array([robot_world_x,robot_world_y,robot_world_z])
            obstacle_pose = None
            start_pose = None
            if self.obs_pose is not None:
                obstacle_pose = robot_world_R@self.obs_pose+robot_pos # np.array([robot_world_x,robot_world_y,robot_world_z])
            if self.start_pose is not None:
                start_pose = robot_world_R@self.start_pose+robot_pos
            if self.approach:
                self.start_pose = None
                goal_pose = robot_world_R@self.goal_pose+robot_pos # np.array([robot_world_x,robot_world_y,robot_world_z])
            else:
                goal_pose = self.start_position[:2]
    
            robot_pose = robot_world_R@robot_pos

            if (self.set_start_position):
                self.start_position = (robot_pose).copy()

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None,None,None
        
        return start_pose, obstacle_pose, goal_pose
    
    def timer_update(self):
        self.get_logger().info("Timer")
        ################### Write your code here ###################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly 
        if (not self.approach and self.step_count >= len(self.steps)):
            self.get_logger().info("END")
            self.pub_control_cmd.publish(Twist())
            return       
        if (self.approach and self.goal_pose is None):
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 1.0
            self.pub_control_cmd.publish(cmd_vel)
            self.get_logger().info("NO GOAL")
            return
        
        # Get the current object pose in the robot base_footprint frame
        current_start_pose, current_obs_pose, current_goal_pose = self.get_current_poses()
        
        cmd_vel = Twist()

        if self.approach:
            # self.get_logger().info(f"goal: {current_goal_pose[:2]}")
            if np.linalg.norm(current_goal_pose) <= 0.35:
                self.approach = False
                self.get_logger().info("Goal Achieved.\n"*20)
                self.pub_control_cmd.publish(Twist())
                return

            # TODO: get the control velocity command
        cmd_vel = self.controller(current_start_pose, current_obs_pose, current_goal_pose)
            # elif self.start_pose is not None:
                # self.get_logger().info(f"start: {current_start_pose[:2]}")
            # if current_obs_pose is not None:
            #     self.get_logger().info(f"obj: {current_obs_pose[:2]}")
        
        self.get_logger().info(f"{len(self.steps)} ; {self.step_count}")
    
        # self.get_logger().info(f"vel: {cmd_vel}")
        
        if not (isinstance(cmd_vel, Twist)):
            self.get_logger().info("NOT A TWIST MSG")
            return

        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self, start_pose, obj_pose, goal_pose):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########
        cmd_vel = Twist()
        if (goal_pose is None and self.approach):
            self.get_logger().info("Goal is None. NO TARGET")
            return cmd_vel
        
        if not self.approach and self.step_count is not None:
            oldx, oldy = self.steps[len(self.steps)-1-round(self.step_count)]
            cmd_vel.linear.x = -oldx
            cmd_vel.linear.y = -oldy
            self.step_count += 0.9
            return cmd_vel
        elif not self.approach and len(self.steps) <= 0:
            return cmd_vel

        scale1 = 1.0
        scale2 = 1.0

        robot_pose = np.zeros(2)
        if self.approach: # The goal pose is relative to robot camera
            target_pose = goal_pose

        else:
            target_pose = start_pose

        if np.linalg.norm(target_pose) < 0.3:
            return cmd_vel

        attractive_str = 0.5*scale1*((np.linalg.norm(target_pose[:2]-robot_pose[:2]))**2)
        attractive_direction = scale1*(target_pose[:2]-robot_pose[:2])

        repulsive_direction = np.zeros(2)
        repulsive_str = 0

        if self.obs_pose is not None:
            EPSILON = 1e-6
            FIELD = 0.5
            d_q = max(np.linalg.norm(obj_pose[:2] - robot_pose[:2]), EPSILON) # prevent divide by zero
            if d_q < FIELD:
                repulsive_str += 0.5*scale2*(((1 / d_q)-(1 / FIELD))**2)
                g_dq = (obj_pose[:2] - robot_pose[:2]) / d_q
                repulsive_direction += 0.5*scale2*((1 / FIELD)-(1 / d_q)) * (1 / (d_q**2)) * g_dq

        total_direction = (attractive_direction+repulsive_direction) / np.linalg.norm((attractive_direction+repulsive_direction))
        total_field = attractive_str+repulsive_str

        K_V = 1.0
        MAX_SPEED = 5.0

        strength = np.clip(total_field, -MAX_SPEED, MAX_SPEED)

        # self.get_logger().info(f"pose: {robot_pose[:2]}, goal: {goal_pose[:2]} dir: {total_direction}, mag: {strength}")

        vel_x = K_V*strength*total_direction[0]
        vel_y = K_V*strength*total_direction[1]
        
        # TODO: Update the control velocity command
        cmd_vel.linear.x = vel_x
        cmd_vel.linear.y = vel_y
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0
        self.steps.append((cmd_vel.linear.x, cmd_vel.linear.y))
        return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
