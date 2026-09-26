import os, sys

import environments

# import python packages
import time
from copy import deepcopy

# import third-party packages
import numpy as np
import torch
import torch.optim as optim
import multiprocessing as mp
import pickle
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent

# import our packages
from scalarization_methods import WeightedSumScalarization
from sample import Sample
from task import Task
from ep import EP
from utils import generate_weights_batch_dfs, print_info, generate_w_batch_test, compute_eu, compute_sparsity
from warm_up import initialize_warm_up_batch
from mopg import MOPG_worker, Extension_worker

from pymoo.factory import get_performance_indicator
#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def eval(eval_sample, ref, obj_num, eval_delta_weight):
    eval_sample = np.asarray(eval_sample, dtype=np.float64)
    if eval_sample.size == 0:
        return 0.0, 0.0, 0.0
    if eval_sample.ndim == 1:
        eval_sample = eval_sample.reshape(1, -1)
    perf_ind = get_performance_indicator("hv", ref_point = -ref)
    preferences = generate_w_batch_test(obj_num, eval_delta_weight)
    hv = perf_ind.do(-eval_sample)
    eu = compute_eu(eval_sample, preferences)
    sp = compute_sparsity(eval_sample)
    return hv, eu, sp


def _mp_context(args):
    env_name = str(getattr(args, "env_name", ""))
    start_method = "spawn" if env_name.startswith("chlor_alkali") else "fork"
    try:
        return mp.get_context(start_method)
    except ValueError:
        return mp.get_context()


def _use_serial_workers(args):
    env_name = str(getattr(args, "env_name", "")).lower()
    return env_name.startswith("chlor_alkali") or ("chlor" in env_name and "alkali" in env_name)

def run(args):

    # --------------------> Preparation <-------------------- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    start_time = time.time()

    # initialize ep and population and opt_graph
    ref = np.array(args.ref_point)
    ep = EP(ref, args.num_select, args.policy_buffer)
    
    # Construct tasks for warm up
    selected_batch, scalarization_batch = initialize_warm_up_batch(args, device)
    
    print_info('\n------------------------------- Initialization Stage -------------------------------')    

    # --------------------> RL Optimization <-------------------- #
    # compose task for each selected solution
    iteration = 0
    task_batch = []
    for solutions, scalarization in \
            zip(selected_batch, scalarization_batch):
        task_batch.append(Task(solutions, scalarization)) # each task is a (policy, weight) pair
    worker_count = len(task_batch)
        
    # Compute update steps for initialization
    num_update = int(args.num_init_steps // len(task_batch)// args.num_steps)

    # run MOPG for each task in parallel
    serial_workers = _use_serial_workers(args)
    if serial_workers:
        processes = []
        results_queue = ThreadQueue()
        done_event = ThreadEvent()
        done_event.set()
        all_offspring_batch = [[] for _ in range(len(task_batch))]
        for task_id, task in enumerate(task_batch):
            MOPG_worker(
                args,
                task_id,
                task,
                device,
                iteration,
                num_update,
                start_time,
                results_queue,
                done_event,
            )
            rl_results = results_queue.get()
            offsprings = rl_results["offspring_batch"]
            for sample in offsprings:
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
    else:
        ctx = _mp_context(args)
        processes = []
        results_queue = ctx.Queue()
        done_event = ctx.Event()

        for task_id, task in enumerate(task_batch):
            p = ctx.Process(target = MOPG_worker, \
                args = (args, task_id, task, device, iteration, num_update, start_time, results_queue, done_event))
            p.start()
            processes.append(p)

        # collect MOPG results for offsprings and insert objs into objs buffer
        all_offspring_batch = [[] for _ in range(worker_count)]
        cnt_done_workers = 0
        while cnt_done_workers < len(processes):
            rl_results = results_queue.get()
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            for sample in offsprings:
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            if rl_results['done']:
                cnt_done_workers += 1
        
    # put all intermidiate policies into all_sample_batch for EP update
    all_sample_batch = [] 
    for task_id in range(worker_count):
        offsprings = all_offspring_batch[task_id]
        for i, sample in enumerate(offsprings):
            all_sample_batch.append(sample)

    if not serial_workers:
        done_event.set()
    iteration += num_update

    # -----------------------> Update EP <----------------------- #
    # update EP and population
    ep.update(all_sample_batch)
    all_sample_batch = []

    # ------------------- > Task Selection <--------------------- #
    selected_batch = ep.selected_batch
        
    print_info('Selected Tasks:')
    for i in range(len(selected_batch)):
        print_info('objs = {}'.format(selected_batch[i].objs))
        
    # save
    obj_array = np.array(ep.obj_batch)
    hv, eu, sp = eval(obj_array, ref, args.obj_num, args.eval_delta_weight)
    print(f"Hyper Volume: {hv:.4f}, Expected Utility: {eu:.4f}, Sparsity: {sp:.4f}")
    
    # ----------------------> Save Results <---------------------- #
    # save ep
    ep_dir = os.path.join(args.save_dir, 'init', 'ep')
    os.makedirs(ep_dir, exist_ok = True)
    with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
        for obj in ep.obj_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))
    
    # save selections
    selected_dir = os.path.join(args.save_dir, 'init', 'solutions')
    os.makedirs(selected_dir, exist_ok = True)
    with open(os.path.join(selected_dir, 'solutions.txt'), 'w') as fp:
        for solutions in selected_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(solutions.objs)))
    with open(os.path.join(selected_dir, 'weights.txt'), 'w') as fp:
        for scalarization in scalarization_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
    

    print_info('\n------------------------------- Extension Stage -------------------------------') 
    
    total_num_update = int((args.num_time_steps - args.num_init_steps) // args.num_select // args.num_steps // args.obj_num)
    extension_iter = total_num_update // args.update_iter
    beta = args.beta
    iteration = 0

    for iter_ in range (extension_iter+1):   
        extension_start_time = time.time()
        num_update = min(args.update_iter, total_num_update - iteration)
        print(f"Starting the extension from iteration {iteration} to {iteration+num_update}.")

        # run MOPG for each task in parallel
        serial_workers = _use_serial_workers(args)
        worker_count = len(selected_batch)
        if serial_workers:
            processes = []
            results_queue = ThreadQueue()
            done_event = ThreadEvent()
            done_event.set()
            all_offspring_batch = [[] for _ in range(len(selected_batch))]
            for sample_id, sample in enumerate(selected_batch):
                Extension_worker(
                    args,
                    sample_id,
                    sample,
                    device,
                    iteration,
                    num_update,
                    extension_start_time,
                    results_queue,
                    done_event,
                )
                rl_results = results_queue.get()
                offsprings = rl_results["offspring_batch"]
                for child_sample in offsprings:
                    all_offspring_batch[sample_id].append(Sample.copy_from(child_sample))
        else:
            ctx = _mp_context(args)
            processes = []
            results_queue = ctx.Queue()
            done_event = ctx.Event()

            for sample_id, sample in enumerate(selected_batch):
                p = ctx.Process(target = Extension_worker, \
                    args = (args, sample_id, sample, device, iteration, num_update, extension_start_time, results_queue, done_event))
                p.start()
                processes.append(p)

            # collect MOPG results for offsprings and insert objs into objs buffer
            all_offspring_batch = [[] for _ in range(worker_count)]
            cnt_done_workers = 0
            while cnt_done_workers < len(processes):
                rl_results = results_queue.get()
                task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
                for sample in offsprings:
                    all_offspring_batch[task_id].append(Sample.copy_from(sample))
                if rl_results['done']:
                    cnt_done_workers += 1

        # put all intermidiate policies into all_sample_batch for EP update
        # all_sample_batch = [] 
        # store the last policy for each optimization weight for RA
        last_offspring_batch = [None] * worker_count
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        offspring_batch = [] 
        for task_id in range(worker_count):
            offsprings = all_offspring_batch[task_id]
            #prev_node_id = task_batch[task_id].sample.optgraph_id
            #opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % args.update_iter == 0:
                    #prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    #sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]

        if not serial_workers:
            done_event.set()
        iteration += num_update

        # -----------------------> Update EP <----------------------- #
        # update EP and population
        ep.update(all_sample_batch)
        all_sample_batch = []

        # ------------------- > Task Selection <--------------------- #
        selected_batch = ep.selected_batch
        print_info('Selected Tasks:')
        for i in range(len(selected_batch)):
            print_info('objs = {}'.format(selected_batch[i].objs))

        # save
        reward_list_array = np.array(ep.obj_batch)
        hv, eu, sp = eval(reward_list_array, ref, args.obj_num, args.eval_delta_weight)
        print(f"Hyper Volume: {hv:.4f}, Expected Utility: {eu:.4f}, Sparsity: {sp:.4f}")

        # ----------------------> Save Results <---------------------- #
        # save ep
        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # save selected solutions
        selected_dir = os.path.join(args.save_dir, str(iteration), 'solutions')
        os.makedirs(selected_dir, exist_ok = True)
        with open(os.path.join(selected_dir, 'solutions.txt'), 'w') as fp:
            for solutions in selected_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(solutions.objs)))
        with open(os.path.join(selected_dir, 'weights.txt'), 'w') as fp:
            for scalarization in scalarization_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
        
    end_time = time.time()
    print('total time:', end_time - start_time)
    
     # ----------------------> Save Final Model <---------------------- 

    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)

    # save ep policies & env_params
    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
        with open(os.path.join(args.save_dir, 'final', 'EP_env_params_{}.pkl'.format(i)), 'wb') as fp:
            pickle.dump(sample.env_params, fp)
    
    # save all ep objectives
    with open(os.path.join(args.save_dir, 'final', 'objs.txt'), 'w') as fp:
        for i, obj in enumerate(ep.obj_batch):
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(obj)))

    # save all ep env_params
    if args.obj_rms:
        with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
            for sample in ep.sample_batch:
                fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))
