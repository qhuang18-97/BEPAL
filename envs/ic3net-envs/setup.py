from setuptools import setup

# Pinned to match envs/rware-envs: gym 0.22+ changes the reset()/step()
# signatures that env_wrappers.GymWrapper unpacks.
#
# NOTE: gym 0.21.0 ships malformed metadata ("opencv-python (>=3.)"), which
# pip >= 24.1 refuses to parse -- both for gym itself and for anything that
# depends on it. Install pip < 24.1 first:
#     pip install "pip<24.1"
#     pip install -e envs/ic3net-envs
setup(name='ic3net_envs',
      version='0.0.1',
      packages=['ic3net_envs'],
      install_requires=[
          'gym==0.21.0',
          'numpy==1.24.3',
          'torch==2.0.1',
          'visdom==0.2.4',
      ],
)
