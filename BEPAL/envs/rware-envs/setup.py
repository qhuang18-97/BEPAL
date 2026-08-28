from setuptools import setup

# Pinned deliberately. rware 1.0.3 declares `gym>=0.20` with no upper bound,
# but gym 0.22+ changes the reset()/step() signatures that rware_env.py and
# env_wrappers.py unpack, so gym must be held at 0.21.0.
#
# NOTE: gym 0.21.0 ships malformed metadata ("opencv-python (>=3.)"), which
# pip >= 24.1 refuses to parse -- both for gym itself and for anything that
# depends on it. Install pip < 24.1 first:
#     pip install "pip<24.1"
#     pip install -e envs/rware-envs
setup(name='rware_envs',
      version='0.0.1',
      packages=['rware_envs'],
      install_requires=[
          'gym==0.21.0',
          'numpy==1.24.3',
          'rware==1.0.3',
          'torch==2.0.1',
          'visdom==0.2.4',
      ],
      # Only needed for the commented-out belief-map plotting in trainer.py.
      extras_require={'plots': ['matplotlib==3.7.5']},
)
