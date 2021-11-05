from envs.motion_generator import MotionGenerator
from IPython import embed

motion_generator = MotionGenerator('./data/bvh/walk.txt')
pose = motion_generator.generate_single(0.1)
embed()
