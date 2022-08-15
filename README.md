# Openai_ROS

Contains Gym environments for interfacing with ROS and Gazebo

## Install Dependencies

Execute the following commands:
        
        cd ~/catkin_ws/src
        git clone git@github.com:CUSail-Navigation/openai_ros.git
        cd ~/catkin_ws
        catkin_make_isolated --install -j1
        rosdep install openai_ros

Make soft links from the install location to the robot_envs and task_envs directories:

        cd ~/catkin_ws/install_isolated/lib/python3/dist-packages/openai_ros
        ln -s ~/catkin_ws/src/openai_ros/openai_ros/src/openai_ros/robot_envs
        ln -s ~/catkin_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs

