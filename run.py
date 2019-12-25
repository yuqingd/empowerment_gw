import numpy as np
from envs.gw_blocks import GridWorldEnv

from emp_by_counting import EmpowermentCountingPolicy
from emp_by_BA import EmpowermentBAPolicy

def run_gridworld_counting_policy(account_for_human, goal_oriented):
    env = GridWorldEnv(seed=1, grid_size=5)
    policy = EmpowermentCountingPolicy(env, horizon=7, num_traj=1000, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)
    s = env.reset()
    num_steps = 10
    env.render()
    for _ in range(num_steps):
        s = env.step_human(s)
        action = policy.next_action(s)
        env.set_state(s)
        next_s, _, _, _ = env.step(action)
        env.render()
        s = next_s


if __name__ == '__main__':
     print("Counting Policy")
     run_gridworld_counting_policy(False, False)

     print("With Human")
     run_gridworld_counting_policy(True, False)

     print("With Human Goal")
     run_gridworld_counting_policy(True, True)