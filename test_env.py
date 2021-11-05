from yaml import parse
from utils.config import *
from envs.humanoid import Humanoid
from envs.ant import Ant
from isaacgym import gymapi
import time







if __name__ == '__main__':
    args = get_args()
    cfg, cfg_train, log_dir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)

    env = Humanoid(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda', 0, True)

    start = time.time()
    for i in range(1):
        #env.set_pose()
        action = torch.Tensor(np.random.normal(0,0.1, (1, 28-0)))
        env.step(action)
        #env.render(True)
    end = time.time()
    print(end -start)