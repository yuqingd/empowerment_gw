import numpy as np
from envs.gw_blocks import GridWorldEnv

from emp_by_counting import EmpowermentCountingPolicy
from emp_by_BA import EmpowermentBAPolicy

def run_gridworld_counting_policy(account_for_human, goal_oriented, test_case, grid_size=5, test_case='center', num_boxes=5):

    #Initialize env and boxes location
    if "center" in test_case:
        center_coord = int(grid_size/2)
        assert center_coord > 0, "Grid too small"
        human_pos=[center_coord,center_coord]
        boxes_pos=[center_coord,center_coord+1]
        boxes_pos+=[center_coord+1, center_coord]
        boxes_pos+=[center_coord, center_coord-1]
        boxes_pos+=[center_coord-1,center_coord]

    elif "corner" in test_case:
        #Randomly choose a corner
        corner = np.random.randint(0,grid_size-1)
        if corner is 0:
            human_pos=[0,0], boxes_pos=[0,1,1,0]
        elif corner is 1:
            human_pos=[0,grid_size-1], boxes_pos=[0,grid_size-2,1,grid_size-1]
        elif corner is 2:
            human_pos=[grid_size-1,0], boxes_pos=[grid_size-2,0,grid_size-1,1]
        elif corner is 3:
            human_pos=[grid_size-1,grid_size-1], boxes_pos=[grid_size-2,grid_size-1,grid_size-1,grid_size-2]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    human_goal = np.random.randint(0,grid_size-1)

    # create env
    env = GridWorldEnv(grid_size, seed=1, p=1.0, human_pos, boxes_pos, human_goal)

    policy = EmpowermentCountingPolicy(env, horizon=10, num_traj=1000, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)

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
    parser = argparse.ArgumentParser(description='Gridworld for Empowerment')
    parser.add_argument('--account_for_human', action='store_true', help='Compute human empowerment')
    parser.add_argument('--goal_oriented', action='store_true', help='Human is goal-oriented')
    parser.add_argument('--grid_size', type=int, default=5, help='Human is goal-oriented')
    parser.add_argument('--test_case', type=str, default='', help='Test Case -- \'center\' for human in center, \'corner\' for human in corner, default is random')
    parser.add_argument('--num_boxes', type=int, default=5, help='Number of boxes in scene')


    args = parser.parse_args()

    account_for_human = args.account_for_human
    goal_oriented = args.goal_oriented
    grid_size = args.grid_size
    test_case = args.test_case
    num_boxes = args.num_boxes

    run_gridworld_counting_policy(account_for_human, goal_oriented, grid_size, test_case, num_boxes)