from setuptools import find_packages, setup

package_name = 'rm_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admin-mrahman',
    maintainer_email='muhammad.ibrahim.rahman@iml.fraunhofer.de',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'publisher = rm_publisher.target_vels_publisher:main',
        ],
    },
)
