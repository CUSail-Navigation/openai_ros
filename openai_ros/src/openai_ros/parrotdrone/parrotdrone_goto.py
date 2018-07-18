import rospy
import numpy
from gym import spaces
from openai_ros import parrotdrone_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3

timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='ParrotDroneGoto-v0',
        entry_point='parrotdrone_goto:ParrotDroneGotoEnv',
        timestep_limit=timestep_limit_per_episode,
    )

class ParrotDroneGotoEnv(parrotdrone_env.droneEnv):
    def __init__(self):
        """
        Make parrotdrone learn how to navigate to get to a point
        """
        
        # Only variable needed to be set here
        number_actions = rospy.get_param('/drone/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/drone/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/drone/angular_turn_speed')
        self.angular_speed = rospy.get_param('/drone/angular_speed')
        
        
        self.init_linear_forward_speed = rospy.get_param('/drone/init_linear_forward_speed')
        
        self.linear_speed_vector = Vector3()
        self.linear_speed_vector.x = rospy.get_param('/drone/linear_speed_vector/x')
        self.linear_speed_vector.y = rospy.get_param('/drone/linear_speed_vector/y')
        self.linear_speed_vector.z = rospy.get_param('/drone/linear_speed_vector/z')
        
        self.init_angular_turn_speed = rospy.get_param('/drone/init_angular_turn_speed')
        

        self.min_sonar_value = rospy.get_param('/drone/min_sonar_value')
        
        
        
        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/drone/work_space/z_min")
        
        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/desired_pose/x")
        self.desired_point.y = rospy.get_param("/drone/desired_pose/y")
        self.desired_point.z = rospy.get_param("/drone/desired_pose/z")
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        # TODO: OBESRVATION SPACE
        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.closer_to_point_reward = rospy.get_param("/drone/closer_to_point_reward")
        self.not_ending_point_reward = rospy.get_param("/drone/not_ending_point_reward")
        self.end_episode_points = rospy.get_param("/drone/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ParrotDroneGotoEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init lienar and angular speeds
        """
        
        # We Issue the landing command to be sure it starts landing
        self.land()
        
        # We TakeOff before sending any movement commands
        self.takeoff()
        
        self.move_base(self.linear_speed_vector,
                        self.init_angular_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.position)


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.angular_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.angular_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        
        # We tell drone the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        discretized_laser_scan = self.discretize_observation( laser_scan,
                                                                self.new_ranges
                                                                )
                                                                
                                                                
        # We get the odometry so that SumitXL knows where it is.
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        odometry_array = [round(x_position, 2),round(y_position, 2)]

        # We only want the X and Y position and the Yaw

        observations = discretized_laser_scan + odometry_array

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("drone is Too Close to wall==>")
        else:
            rospy.logerr("drone didnt crash at least ==>")
       
       
            current_position = Point()
            current_position.x = observations[-2]
            current_position.y = observations[-1]
            current_position.z = 0.0
            
            MAX_X = 6.0
            MIN_X = -1.0
            MAX_Y = 3.0
            MIN_Y = -3.0
            
            # We see if we are outside the Learning Space
            
            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("TurtleBot Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")
                    
                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done = True
                    
                    
                else:
                    rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(current_position.x))
                    self._episode_done = True
            else:
                rospy.logerr("TurtleBot to Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True
            
            
            

        return self._episode_done

    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[-2]
        current_position.y = observations[-1]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point


        if not done:
            
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
                
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.forwards_reward
            else:
                rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward += 0
                
        else:
            
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points


        self.previous_distance_from_des_point = distance_from_des_point


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))
                    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    

        return discretized_ranges
        
        
    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos
        
        
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
    
        distance = numpy.linalg.norm(a - b)
    
        return distance
