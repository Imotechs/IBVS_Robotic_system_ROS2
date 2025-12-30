from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_bringup'

def package_files(base_dir):
    paths = []
    for (path, _, filenames) in os.walk(base_dir):
        for filename in filenames:
            full_path = os.path.join(path, filename)
            relative_path = os.path.relpath(path, base_dir)
            install_path = os.path.join('share', package_name, base_dir, relative_path) if relative_path != '.' \
                else os.path.join('share', package_name, base_dir)
            paths.append((install_path, [full_path]))
    return paths


# Collect model files while preserving folder layout
model_data_files = package_files('models')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch, URDF, Configs
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),

        # Materials
        (os.path.join('share', package_name, 'materials/textures'), glob('materials/textures/*')),
        (os.path.join('share', package_name, 'materials/scripts'), glob('materials/scripts/*')),
        # ✅ Properly include only files from models directory
        *model_data_files,  # ✅ include all model files recursively, preserving hierarchy

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Robot bringup package with YOLOv5-based QR detection and Gazebo integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_view = robot_bringup.camera_view:main',
            'move_arm = robot_bringup.move_arm:main',
            'conveyor_controller = robot_bringup.conveyor_control:main',
            'move_product = robot_bringup.move_products:main',
            'ibvs_control = robot_bringup.ibvs_control:main',
            'gripper_node = robot_bringup.gripper:main',
            'dashboard = robot_bringup.dashboard:main',

        ],
    },
)
