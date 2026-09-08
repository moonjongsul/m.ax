from glob import glob
from setuptools import find_packages, setup

package_name = 'max_data_collect'

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
    description=(
        'Records teleop demonstrations and VLA rollouts as per-episode '
        'HDF5 + one mp4 per camera.'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'max_data_collect = max_data_collect.recorder_node:main',
        ],
    },
)
