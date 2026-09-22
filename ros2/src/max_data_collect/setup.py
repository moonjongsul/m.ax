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
    # The editor's web assets are data, not modules; without this they are
    # dropped from an installed package and the app serves a blank page.
    package_data={package_name + '.editor': ['static/*', 'README.md']},
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='keti10829',
    maintainer_email='moonjongsul@gmail.com',
    description=(
        'Records teleop demonstrations and VLA rollouts as per-episode '
        'HDF5 + one mp4 per camera, and curates them for VLA training.'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'max_data_collect = max_data_collect.recorder_node:main',
            'max_data_editor = max_data_collect.editor.server:main',
        ],
    },
)
