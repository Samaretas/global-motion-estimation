import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(name='global_motion_estimation',
                 packages=['global_motion_estimation'],
                 install_requires=install_requires)
