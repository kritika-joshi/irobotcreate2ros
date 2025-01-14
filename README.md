irobotcreate2ros
===
Ros node for iRobot Create 2.


Updated Procedures for MECH 464 iRobot Create2 team, Winter 2021/22.

Creating the base ROS workspace:
- Start by creating a workspace in your local folder as follows: mkdir -p mech464_ws/src
- cd mech464_ws
- catkin_make
- catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
- source devel/setup.bash
- echo $ROS_PACKAGE_PATH /home/fizzer/mech464_ws/src:/opt/ros/melodic/share

Add ros_serial to your src:
- cd /home/fizzer/mech464_ws/src
- git clone git://github.com/wjwwood/serial.git
- cd serial
- make

Adding the irobotcreate2ros node to your folder:
- cd /home/fizzer/mech464_ws/src
- git clone https://github.com/kritika-joshi/irobotcreate2ros.git
- cd ..
- catkin_make
- cd src/irobotcreate2ros/model
- source generate_model.sh
- cd
- subl .bashrc
- add the following commands to your bashrc: 
  - source /usr/share/gazebo/setup.sh
  - export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find irobotcreate2)
  - export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(rospack find irobotcreate2)
  - source /home/fizzer/mech464_ws/devel/setup.bash

Running gmapping - TEMP:
- rosrun gmapping slam_gmapping scan:=/iRobot/laser_scan
- rosrun tf static_ansform_publisher 0 0 0 0 0 0 /base_link /iRobot/hokuyo_laser_link 10
- switch to Rviz and add Map from "add topic", and set it to be wrt /map

Add your user to dialout group so it connects to USB properly:
- sudo usermod -a -G dialout fizzer
- log out and log in again

Running with the physical robot:
- cd /home/fizzer/mech464_ws/src/irobotcreate2ros/
- roslaunch irobotcreate2 irobotcreate2.launch
- publish command velocities in a different terminal as follows:
  - rostopic pub -r 10 /iRobot_0/cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}' 

Running the robot on Gazebo:
- cd /home/fizzer/mech464_ws/src/irobotcreate2ros/
- roslaunch irobotcreate2 gazebo.launch
- publish command velocities in a different terminal as follows:
  - rostopic pub /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}' 

Teleoperation (Manual Keyboard Controls)
- rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/iRobot_0/cmd_vel
- Type x to reduce speed
- Make sure speed is reasonable before starting (changing speed also triggers a small movement)
- Use "i" to move forwards, "<" to move backwards, "l" to rotate clockwise, and "j" to rotate anti-clockwise
- There also buttons for moving in an arc
- Pressing the button once causes a jolt but holding the button will cause continuous motion

Bugs - if you don't have catkin_make installed:
sudo apt-get install ros-melodic-catkin python-catkin-tools

Prerequisites
---
* [ros serial](http://wiki.ros.org/serial)
* [ros joy](http://wiki.ros.org/joy)

Compiling
---
Just clone the repository in the src folder of a catkin workspace. Then run catkin_make.

To generate the model you have to launch in the model folder the dedicated script:
```
source generate_model.sh
```

To simulate the robot in Gazebo you first have to add this lines to your bashrc:

```
source /usr/share/gazebo/setup.sh
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find irobotcreate2)
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(rospack find irobotcreate2)
```

Usage
---
---
```
rosrun irobotcreate2 irobotcreate2
```
to run the basic software and have access to the following topics:

- /battery
- /bumper
- /buttons
- /cliff
- /cmd_vel
- /digit_leds
- /ir_bumper
- /ir_character
- /leds
- /mode
- /odom
- /play_song
- /rosout
- /rosout_agg
- /song
- /tf
- /wheel_drop

you can read sensors (/battery, /buttons, /bumper, ...) and send commands (/cmd_vel, ...).

---
```
roslaunch irobotcreate2 irobot_joy.launch
```
to run both the basic software and ros joy to move the robot with a controller.

<!--[Instructions](cad/laser_support/instructions.md)-->

