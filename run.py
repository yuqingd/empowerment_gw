import numpy as np
from envs.gw_blocks import GridWorldEnv
import argparse
from datetime import datetime
import os
from emp_by_counting import EmpowermentCountingPolicy
from goal_inference import GoalInferencePolicy

def rollout_emp(env, policy, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=False, goal_oriented=False, goal_inference=False):
    if policy is None:
        policy = EmpowermentCountingPolicy(env, horizon=10, num_traj=100, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)

    s = env.reset()
    num_steps = 1000

    filename =  results_folder + "/trial_num_" + str(trial_num)
    if account_for_human:
       filename+="_account_for_human"
    if goal_oriented:
       filename+="_goal_oriented"
    if goal_inference:
       filename+="_goal_inference"
    filename+=".txt"

    file = open(filename, "w")
    file.write("----------------- NEW EXPERIMENT ----------------")
    file.close()
    env.render(filename)
    env.render()

    for step in range(num_steps):
        s_prev = s
        s, done, h_ac = env.step_human(s)
        if done:
            file = open(filename, "a")
            file.write("Reached Goal, took {} steps".format(step))
            file.close()
            break
        if goal_inference:
            action = policy.next_action(s, s_prev)
            env.set_state(s)
            if action >= 0: #don't step if -1
                try:
                    next_s, _, _, _ = env.step(action)
                except:
                    import pdb; pdb.set_trace()
        else:
            action = policy.next_action(s)
            env.set_state(s)
            next_s, _, _, _ = env.step(action)
        env.render(filename)
        env.render()
        s = next_s

    return step




def run_gridworld_counting_policy(goal_inference, account_for_human, goal_oriented, test_all, results_folder, trial_num, test_case='center', grid_size=5, goal_num='all', include_goal=True,num_boxes=5, block_goal=True):
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
    elif "random" in test_case:
        #random location
        occupied_locs = set()
        human_pos = np.random.randint(0,grid_size-1, 2)
        occupied_locs.add(tuple(human_pos))
        boxes_pos = []
        while len(boxes_pos) < num_boxes * 2:
            pos = np.random.randint(0,grid_size-1, 2)
            if tuple(pos) in occupied_locs:
                continue
            else:
                boxes_pos.extend(pos)
                occupied_locs.add(tuple(pos))
    else:
        raise NotImplementedError


    #Initialize human goal position
    coords = np.arange(grid_size)
    all_coords = np.array(np.meshgrid(coords, coords)).T.reshape(-1,2) #TODO: Check this generates all coordinates
    goal_set = set()
    if block_goal:
        #goal can be covered by blocks
        human_goal = np.random.randint(0,grid_size-1, 2)
    else:
        boxes_coords = np.reshape(boxes_pos, (num_boxes, 2))
        while True:
            human_goal = np.random.randint(0,grid_size-1, 2)
            if ((boxes_coords == human_goal).all(axis=1)).any():
                continue
            else:
                break


    if goal_num == 'all':
        for coord in all_coords:
                # test case where goal is not in goal set
            if not include_goal and np.array_equal(coord, human_goal):
                continue
            goal_set.add(tuple(coord))
    else:
        human_goal_idx = np.where((all_coords == human_goal).all(axis=1))[0]
        if not include_goal:
            goals_idx = np.random.choice(np.delete(all_coords, human_goal_idx, axis=0).shape[0], grid_size//2, replace=False)
        else:
             goals_idx = np.random.choice(all_coords.shape[0], grid_size//2 - 1, replace=False)
             goals_idx = np.concatenate([goals_idx, human_goal_idx])

        for idx in goals_idx:
            coord = all_coords[idx]
            goal_set.add(tuple(coord))





    # create env
    env = GridWorldEnv(seed=1, grid_size=grid_size, p=1.0, human_pos=human_pos, boxes_pos=boxes_pos, human_goal=human_goal)

    if test_all:
        step_s1 = rollout_emp(env, None, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=False, goal_oriented=False)
        step_s2 = rollout_emp(env, None, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=True, goal_oriented=False)
        step_s3 = rollout_emp(env, None, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=True, goal_oriented=True)
        return [step_s1, step_s2, step_s3]
    elif goal_inference:
        policy_gi = GoalInferencePolicy(env, goal_set)
        return rollout_emp(env, policy_gi, results_folder, trial_num, horizon=10, num_traj=100, goal_inference=True)
    else:
        policy_emp = EmpowermentCountingPolicy(env, horizon=10, num_traj=100, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)
        return rollout_emp(env, policy_emp, results_folder, trial_num, horizon=10, num_traj=100, estimate_emp=True, account_for_human=account_for_human, goal_oriented=goal_oriented)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gridworld for Empowerment')
    parser.add_argument('--goal_inference', action='store_true', help='Use goal inference policy')
    parser.add_argument('--goal_num', type=str, default='all', help='Goal type: \'all\' for all coordinates can be goals, \'limited\' for goal set that does not contain actual goal')
    parser.add_argument('--include_goal',  action='store_true', help='Include human goal in goal set')
    parser.add_argument('--account_for_human', action='store_true', help='Compute human empowerment')
    parser.add_argument('--goal_oriented', action='store_true', help='Human is goal-oriented')
    parser.add_argument('--test_all', action='store_true', help='Run through s1, s2, s3')
    parser.add_argument('--block_goal', action='store_true', help='Blocks can be on goal')
    parser.add_argument('--grid_size', type=int, default=5, help="Size of grid")
    parser.add_argument('--test_case', type=str, default='random', help='Test Case -- \'center\' for human in center, \'corner\' for human in corner, default is random')
    parser.add_argument('--num_boxes', type=int, default=2, help='Number of boxes in scene')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials')


    args = parser.parse_args()

    account_for_human = args.account_for_human
    goal_oriented = args.goal_oriented
    grid_size = args.grid_size
    test_case = args.test_case
    num_boxes = args.num_boxes
    num_trials = args.num_trials
    test_all = args.test_all
    goal_inference = args.goal_inference
    goal_num = args.goal_num
    include_goal = args.include_goal
    block_goal = args.block_goal

    if not os.path.exists('data'):
        os.makedirs('data')

    now = datetime.now()

    date = now.strftime("%m-%d-%Y-%H-%M-%S")
    results_folder = 'data/'+date+"_"+test_case
    if block_goal:
        results_folder += "_block_goal"
    else:
        results_folder += "_no_block_goal"
    if test_all:
        results_folder += "_test_all"
    elif goal_inference:
        results_folder += "_goal_inference_" + goal_num
        if include_goal:
            results_folder += "_include_goal"
        else:
            results_folder += "_no_include_goal"
    else:
        if account_for_human:
            results_folder+="_account_for_human"
        if goal_oriented:
            results_folder+="_goal_oriented"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    steps = []
    for i in range(num_trials):
        num_steps = run_gridworld_counting_policy(goal_inference, account_for_human, goal_oriented, test_all, results_folder, i, test_case, grid_size, goal_num, include_goal, num_boxes, block_goal)
        steps.append(num_steps)

    filename =  results_folder + "/summary.txt"
    file = open(filename, "w")
    file.write("----------------- SUMMARY ---------------- \n")

    if test_all:
        file.write('Mean steps to goal:' + str(np.mean(steps, axis=0)) + " \n")
        file.write("Median steps to goal: " + str(np.median(steps,axis=0)) + "\n")
        file.write('Std steps to goal:' + str(np.std(steps,axis=0))+ " \n")
        file.write('Max steps to goal:' + str(np.max(steps,axis=0)) + " \n")
        file.write('Min steps to goal:' + str(np.min(steps,axis=0))+ " \n")

    else:
        file.write('Mean steps to goal:' + str(np.mean(steps)) + " \n")
        file.write('Median steps to goal:' + str(np.median(steps))+ " \n")
        file.write('Std steps to goal:' + str(np.std(steps))+ " \n")
        file.write('Max steps to goal:' + str(np.max(steps)) + " \n")
        file.write('Min steps to goal:' + str(np.min(steps))+ " \n")
    file.close()