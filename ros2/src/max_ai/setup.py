from glob import glob

from setuptools import find_packages, setup

package_name = 'max_ai'

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
    zip_safe=False,
    maintainer='keti10829',
    maintainer_email='moonjongsul@gmail.com',
    description=(
        'Runs SmolVLA / MVLA checkpoints on the xArm7 kitting cell as four '
        'nodes: policy inference and control, task manager, planner, and '
        'state recognizer.'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'max_ai_policy = max_ai.nodes.policy_node:main',
            'max_ai_smolvla = max_ai.nodes.smolvla_node:main',
            'max_ai_pi05 = max_ai.nodes.pi05_node:main',
            'max_ai_manager = max_ai.nodes.manager_node:main',
            'max_ai_planner = max_ai.nodes.planner_node:main',
            'max_ai_recognizer = max_ai.nodes.recognizer_node:main',
            'max_ai_replay = max_ai.rollout.replay:main',
        ],
    },
)
