from glob import glob
from setuptools import find_packages, setup

package_name = 'max_server'

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
    maintainer='keti10829',
    maintainer_email='moonjongsul@gmail.com',
    description='Core ROS 2 nodes for the m.ax server (inference, communication, task).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'max_server_node = max_server.max_server_node:main',
        ],
    },
)
