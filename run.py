import numpy as np
from envs.gw_blocks import GridWorldEnv
import argparse
from datetime import datetime
import os
from emp_by_counting import EmpowermentCountingPolicy

def rollout(env, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=False, goal_oriented=False):
    policy = EmpowermentCountingPolicy(env, horizon=10, num_traj=100, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)

    s = env.reset()
    num_steps = 1000

    filename =  results_folder + "/trial_num_" + str(trial_num)
    if account_for_human:
       filename+="_account_for_human"
    if goal_oriented:
       filename+="_goal_oriented"
    filename+=".txt"

    file = open(filename, "w")
    file.write("----------------- NEW EXPERIMENT ----------------")
    file.close()
    env.render(filename)
    env.render()

    for step in range(num_steps):
        s, done = env.step_human(s)
        if done:
            file = open(filename, "a")
            file.write("Reached Goal, took {} steps".format(step))
            file.close()
            break
        action = policy.next_action(s)
        env.set_state(s)
        next_s, _, _, _ = env.step(action)
        env.render(filename)
        env.render()
        s = next_s

    return step

def run_gridworld_counting_policy(account_for_human, goal_oriented, test_all, results_folder, trial_num, test_case='center', grid_size=5):
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
        if corner == 0:
            human_pos=[0,0]
            boxes_pos=[0,1,1,0]
        elif corner == 1:
            human_pos=[0,grid_size-1]
            boxes_pos=[0,grid_size-2,1,grid_size-1]
        elif corner == 2:
            human_pos=[grid_size-1,0]
            boxes_pos=[grid_size-2,0,grid_size-1,1]
        elif corner == 3:
            human_pos=[grid_size-1,grid_size-1]
            boxes_pos=[grid_size-2,grid_size-1,grid_size-1,grid_size-2]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    human_goal = np.random.randint(0,grid_size-1, 2)

    # create env
    env = GridWorldEnv(seed=1, grid_size=grid_size, p=1.0, human_pos=human_pos, boxes_pos=boxes_pos, human_goal=human_goal)

    if test_all:
        step_s1 = rollout(env, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=False, goal_oriented=False)
        step_s2 = rollout(env, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=True, goal_oriented=False)
        step_s3 = rollout(env, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=True, goal_oriented=True)
        return [step_s1, step_s2, step_s3]

    else:
        return rollout(env, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gridworld for Empowerment')
    parser.add_argument('--account_for_human', action='store_true', help='Compute human empowerment')
    parser.add_argument('--goal_oriented', action='store_true', help='Human is goal-oriented')
    parser.add_argument('--test_all', action='store_true', help='Run through s1, s2, s3')
    parser.add_argument('--grid_size', type=int, default=5, help="Size of grid")
    parser.add_argument('--test_case', type=str, default='', help='Test Case -- \'center\' for human in center, \'corner\' for human in corner, default is random')
    parser.add_argument('--num_boxes', type=int, default=5, help='Number of boxes in scene')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials')


    args = parser.parse_args()

    account_for_human = args.account_for_human
    goal_oriented = args.goal_oriented
    grid_size = args.grid_size
    test_case = args.test_case
    num_boxes = args.num_boxes
    num_trials = args.num_trials
    test_all = args.test_all


    if not os.path.exists('data'):
        os.makedirs('data')

    now = datetime.now()

    date = now.strftime("%m-%d-%Y-%H-%M-%S")
    results_folder = 'data/'+date+"_"+test_case
    if test_all:
        results_folder += "_test_all"
    else:
        if account_for_human:
            results_folder+="_account_for_human"
        if goal_oriented:
            results_folder+="_goal_oriented"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    steps = []
    for i in range(num_trials):
        num_steps = run_gridworld_counting_policy(account_for_human, goal_oriented, test_all, results_folder, i, test_case, grid_size)
        steps.append(num_steps)

    filename =  results_folder + "/summary.txt"
    file = open(filename, "w")
    file.write("----------------- SUMMARY ---------------- \n")

    if test_all:
        file.write('Mean steps to goal:' + str(np.mean(steps, axis=0)) + " \n")
        file.write('Std steps to goal:' + str(np.std(steps,axis=0))+ " \n")
        file.write('Max steps to goal:' + str(np.max(steps,axis=0)) + " \n")
        file.write('Min steps to goal:' + str(np.min(steps,axis=0))+ " \n")

    else:
        file.write('Mean steps to goal:' + str(np.mean(steps)) + " \n")
        file.write('Std steps to goal:' + str(np.std(steps))+ " \n")
        file.write('Max steps to goal:' + str(np.max(steps)) + " \n")
        file.write('Min steps to goal:' + str(np.min(steps))+ " \n")
    file.close()