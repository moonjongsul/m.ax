from glob import glob
from setuptools import find_packages, setup

package_name = 'xarm_state_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='moonjongsul@gmail.com',
    description='Cartesian end-effector state (pose and twist) for the xArm',
    license='BSD',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eef_state_publisher = xarm_state_publisher.eef_state_publisher:main',
        ],
    },
)
