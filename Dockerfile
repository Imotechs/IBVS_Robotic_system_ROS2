FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies & core ROS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    python3-numpy \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    udev \
    libopencv-dev \
    git \
    curl \
    wget \
    x11-apps \
    # ROS base tools
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    # ROS2 Control stack
    ros-humble-controller-manager \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ros2-control-test-assets \
    ros-humble-gripper-controllers \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim  \
    ros-humble-controller-manager \
    ros-humble-xacro \ 
    ros-humble-ign-ros2-control \
    # Gazebo ROS integration
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    # MoveIt packages
    ros-humble-moveit \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-setup-assistant \
    ros-humble-moveit-configs-utils \
    # Misc
    ros-humble-trajectory-msgs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir \
    matplotlib \
    pyserial \
    # mediapipe \
    pyrealsense2 \
    "numpy<2.0"

# Setup workspace
WORKDIR /ros2_ws/src


# Clone ros_gz from source (bridge, sim, image, gz_ros2_control, etc.)
RUN git clone -b humble https://github.com/gazebosim/ros_gz.git


# Install ROS dependencies via rosdep
WORKDIR /ros2_ws
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build all packages
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Source environments at startup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=/ros2_ws/src/ridgeback_ur5_gazebo/models' >> ~/.bashrc

CMD ["bash"]

sudo apt install ros-humble-ign-ros2-control
sudo apt install ros-humble-ros-gz-sim-demos

# sudo apt update
# sudo apt install ros-humble-ros2-control ros-humble-controller-manager
# sudo apt update
# sudo apt install ros-humble-gripper-controllers
# sudo apt update
# sudo apt install ros-humble-ros2-controllers
# sudo apt install ros-humble-xacro

# sudo apt update
# sudo apt install ros-humble-ros-gz-sim

# sudo apt update
# sudo apt install ros-humble-ur-description

# sudo apt update
# sudo 
# ros-humble-gz-ros2-control
qrcode with plender :https://www.youtube.com/watch?v=rsD4U9nc_O8

"""
To be added

apt update
apt install ros-humble-cv-bridge python3-opencv

"""


cd /ros2_ws/src
git clone https://github.com/IFRA-Cranfield/IFRA_ConveyorBelt.git
cd /ros2_ws
colcon build --packages-select ifra_conveyorbelt
source install/setup.bash

ros2 service call /conveyor/control ifra_conveyorbelt/srv/SetConveyorPower "{power: 50.0}"


'''
Gazebo Classic 11.10.2 (runs with gazebo)

Ignition Gazebo 6.16.0 (runs with ign gazebo)

sudo apt-get update
sudo apt-get install ros-humble-ros-gz-sim ros-humble-ros-gz-sim-demos
sudo apt install ros-humble-ign-ros2-control
sudo apt install ros-humble-ros-gz-sim-demos
sudo apt install ros-humble-ros-ign-gazebo
sudo apt install ros-humble-gz-ros2-control ros-humble-gz-ros2-control-demos

https://www.linkedin.com/pulse/gazebo-custom-plugin-ros2-step-by-step-guide-shantanu-parab/?trackingId=2iAHO8GATRWbAh1hD%2BvEMQ%3D%3D
'''