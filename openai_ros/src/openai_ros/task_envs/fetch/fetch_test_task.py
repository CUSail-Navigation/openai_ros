from gym import utils
import copy
import rospy
from gym import spaces
from openai_ros.robot_envs import fetch_env
from gym.envs.registration import register
import numpy as np
from sensor_msgs.msg import JointState


register(
        id='FetchTest-v0',
        entry_point='openai_ros:task_envs.fetch.fetch_test_task.FetchTestEnv',
        timestep_limit=50,
    )
    

class FetchTestEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self):
        
        rospy.logdebug("Entered FetchTestEnv Env")
        self.get_params()
        
        fetch_env.FetchEnv.__init__(self)

        
        self.action_space = spaces.Discrete(self.n_actions)
        
        observations_high_range = np.array([self.position_ee_max]*self.n_obervations)
        observations_low_range = np.array([self.position_ee_min]*self.n_obervations)
        self.observation_space = spaces.Box(observations_low_range, observations_high_range)
        

        
    def get_params(self):
        #get configuration parameters
        
        self.n_actions = rospy.get_param('/fetch/n_actions')
        self.n_obervations = rospy.get_param('/fetch/n_obervations')
        self.position_ee_max = rospy.get_param('/fetch/position_ee_max')
        self.position_ee_min = rospy.get_param('/fetch/position_ee_min')
        
        self.init_pos = rospy.get_param('/fetch/init_pos')
        self.setup_ee_pos = rospy.get_param('/fetch/setup_ee_pos')
        self.goal_ee_pos = rospy.get_param('/fetch/goal_ee_pos')
        
        self.position_delta = rospy.get_param('/fetch/position_delta')
        self.step_punishment = rospy.get_param('/fetch/step_punishment')
        self.impossible_movement_punishement = rospy.get_param('/fetch/impossible_movement_punishement')
        self.reached_goal_reward = rospy.get_param('/fetch/reached_goal_reward')
        
        
    
    
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        # Check because it seems its not being used
        rospy.logdebug("Init Pos:")
        rospy.logdebug(initial_qpos)

        
        # Init Joint Pose
        rospy.logerr("Moving To SETUP Joints ")
        self.movement_result = self.set_trajectory_joints(self.init_pos)

        if self.movement_result:
            # INIT POSE
            rospy.logerr("Moving To SETUP Position ")
            self.last_gripper_target = [self.setup_ee_pos["x"],self.setup_ee_pos["y"],self.setup_ee_pos["z"]]
            gripper_rotation = [1., 0., 1., 0.]
            action = self.create_action(gripper_target,gripper_rotation)
            self.movement_result = self.set_trajectory_ee(action)
        
        self.last_action = "INIT"
        
        rospy.logwarn("Init Pose Results ==>"+str(self.movement_result))

        return self.movement_result


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        The simulation will be paused, therefore all the data retrieved has to be 
        from a system that doesnt need the simulation running, like variables where the 
        callbackas have stored last know sesnor data.
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")
    

    def _set_action(self, action):
        
        
        delta_gripper_target = [0.0]*len(self.last_gripper_target)
        
        
        # We convert action ints in increments/decrements of one of the axis XYZ
        if action == 0: # X+
            delta_gripper_target[0] += self.position_delta
            self.last_action = "X+"
        elif action == 1: # X-
            delta_gripper_target[0] -= self.position_delta
            self.last_action = "X-"
        elif action == 2: # Y+
            delta_gripper_target[1] += self.position_delta
            self.last_action = "Y+"
        elif action == 3: # Y-
            delta_gripper_target[1] -= self.position_delta
            self.last_action = "Y-"
        elif action == 4: #Z+
            delta_gripper_target[2] += self.position_delta
            self.last_action = "Z+"
        elif action == 5: #Z-
            delta_gripper_target[2] -= self.position_delta
            self.last_action = "Z-"
        
        
        gripper_target = copy.deepcopy(self.last_gripper_target)
        gripper_target[0] += delta_gripper_target[0]
        gripper_target[1] += delta_gripper_target[1]
        gripper_target[2] += delta_gripper_target[2]
        
        gripper_rotation = [1., 0., 1., 0.]
        # Apply action to simulation.
        action_end_effector = self.create_action(gripper_target,gripper_rotation)
        self.movement_result = self.set_trajectory_ee(action_end_effector)
        if self.movement_result:
            # If the End Effector Positioning was succesfull, we replace the last one with the new one.
            self.last_gripper_target = copy.deepcopy(gripper_target)
        else:
            rospy.logerr("Impossible End Effector Position...."+str(gripper_target))
        
        rospy.logdebug("END Set Action ==>"+str(action)+", NAME="+str(self.last_action))

    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        Orientation for the moment is not considered
        """
        
        grip_pos = self.get_ee_pose()
        grip_pos_array = [grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z]
        obs = grip_pos_array

        return obs
        
    def _is_done(self, observations):
        
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """
        desired_position = [self.goal_ee_pos["x"],self.goal_ee_pos["y"],self.goal_ee_pos["z"]]
        current_pos = observations
        
        done, reward = self.calculate_reward_and_if_done(self.movement_result,desired_position,current_pos)
        
        return done
        
    def _compute_reward(self, observations, done):

        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        
        _ , reward = self.calculate_reward_and_if_done(self.movement_result,desired_position,current_pos)
        
        return reward
        

    def calculate_reward_and_if_done(self, movement_result,desired_position,current_pos):
        """
        It calculated whather it has finished or nota and how much reward to give
        """
        done = False
        reward = self.step_punishment
        
        if movement_result:
            position_similar = np.all(np.isclose(desired_position, current_pos, atol=1e-02))
            if position_similar:
                done = True
                reward = self.reached_goal_reward
                rospy.logerr("Reached a Desired Position!")
        else:
            done = True
            reward = self.crashed_punishment
            rospy.logerr("Reached a TCP position not reachable")
            
        return done, reward