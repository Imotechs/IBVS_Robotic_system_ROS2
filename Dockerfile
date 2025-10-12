# ==========================================================
# Base ROS 2 Humble (Ubuntu 22.04)
# ==========================================================
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# ----------------------------------------------------------
# Install System Dependencies & Core ROS Packages
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    python3-numpy \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libopencv-dev \
    udev \
    git \
    curl \
    wget \
    x11-apps \
    ca-certificates \
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
    ros-humble-ros-gz-sim \
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

# ----------------------------------------------------------
# Install PyTorch (CPU) + Vision Libraries
# ----------------------------------------------------------
    RUN python3 -m pip install --upgrade "pip<25" "setuptools<80" wheel && \
    echo "Installing PyTorch from PyPI..." && \
    pip install --no-cache-dir torch torchvision torchaudio && \
    python3 -c "import torch; print('✅ PyTorch installed OK:', torch.__version__)" && \
    pip install --no-cache-dir \
        matplotlib \
        pyserial \
        pyrealsense2 \
        "numpy<2.0" \
        opencv-python \
        pandas \
        seaborn \
        tqdm \
        ultralytics \
        PyYAML \
        scipy \
        rospkg

# ----------------------------------------------------------
# Setup Workspace
# ----------------------------------------------------------
WORKDIR /ros2_ws/src

# Clone ros_gz (Gazebo integration)
RUN git clone -b humble https://github.com/gazebosim/ros_gz.git

# ----------------------------------------------------------
# Install ROS Dependencies via rosdep
# ----------------------------------------------------------
WORKDIR /ros2_ws
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# ----------------------------------------------------------
# Build the Workspace
# ----------------------------------------------------------
RUN pip install "setuptools<69" && \
    . /opt/ros/humble/setup.sh && colcon build --symlink-install
# ----------------------------------------------------------
# Source Environments at Startup
# ----------------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=/ros2_ws/src/ridgeback_ur5_gazebo/models' >> ~/.bashrc

# ----------------------------------------------------------
# Default Command
# ----------------------------------------------------------
CMD ["bash"]
