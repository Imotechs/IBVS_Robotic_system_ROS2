from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'materials/textures'), glob('materials/textures/*')),
        (os.path.join('share', package_name, 'materials/scripts'), glob('materials/scripts/*')),

        # Install URDF files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # (os.path.join('share', package_name, 'models'), [f for f in glob('models/**/*', recursive=True) if os.path.isfile(f)],),        
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_view = robot_bringup.camera_view:main',
            'move_arm = robot_bringup.move_arm:main',
            'conveyor_controller = robot_bringup.conveyor_control:main',
            'move_product = robot_bringup.move_products:main',

        ],
    },
)
