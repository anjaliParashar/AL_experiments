import sys
import os
import os.path as path
import random
root_dir = path.abspath(path.join(__file__ ,"../../.."))
home_dir = path.abspath(path.join(__file__ ,"../../../../.."))
sys.path.append(path.abspath(path.join(__file__ ,"../../..")))
sys.path.append(path.abspath(path.join(__file__ ,"../../../..")))
from utils import ExpectedCoverageImprovement, get_and_fit_gp#, identify_samples_which_satisfy_constraints
import torch
from botorch.optim import optimize_acqf
from botorch.models import ModelListGP
import argparse
import pickle
import numpy as np

from mit_perception.scripts.inference_functional import run_diffusion_policy


"""
Implements system agent. This is used as a baseline for ground truth, i.e., the best-case performance. 
The true reward is provided. 
"""
def unnormalize(X):
    return (X[0]*350+150,X[1]*300) # Unnormalize [0,1]X[0,1] to [100,400]X[100,400]


def get_user_feedback(X,dim,y_dim,seed,dataset_path,ckpt_path,skip_iter=False,move_gripper=False):
    theta0=45
    x0, y0 = unnormalize(X)
    # x0, y0 = 450,200
    print(f'unnormalized: {(x0,y0)}')

    if not skip_iter:
        record_dict = run_diffusion_policy(x0, y0, theta0, dataset_path, seed, max_steps=200, use_sim=False, move_gripper=move_gripper)
    # print(record_dict['obs_list'][20])
    else:
        print("skipping experiment because you said so -- i assume you already ran it and are resuming or something?")
        record_dict = None

    # score_human = run_diffusion_policy(x0=x0,y0=y0,theta0=np.deg2rad(theta0),seed=seed,dataset_path=dataset_path,ckpt_path=ckpt_path)

    n_modes = y_dim
    while True:
        input_str = input(f"please enter failure mode costs ^-^ : ")
        score_human = input_str.split(',')
        if len(score_human) != n_modes:
            print(f"please enter exactly {n_modes} comma-separated values >_<")
        else:
            break
    score_human = torch.tensor([float(x) for x in score_human]).reshape(1,n_modes)

    return score_human, record_dict

def system_agent(X,dim,y_dim,seed,dataset_path,ckpt_path,skip_iter=False,move_gripper=False):

    scores,record_dict = get_user_feedback(X,dim,y_dim,seed,dataset_path,ckpt_path,skip_iter,move_gripper)
    # if score<0.7:
    #     return torch.tensor(1)
    # else:1,score2,score3
    #     return torch.tensor(1-score)
    # return torch.tensor(score1),torch.tensor(score2),torch.tensor(score3)
    return scores,record_dict

def fail_bo(
        num_iter,bounds,X,Y,
        constraints,dataset_path,ckpt_path,dim,y_dim,punchout_radius,
        record_dicts, resume, lambda_, run_suffix, move_gripper_first):

    if resume:

        print(X, X.shape) 
        print(Y, Y.size())
        print(X.shape, Y.size())

        manually_append = input("do you want to manually append anything to Y? (y/[n]) : ")
        if manually_append == 'y':
            mys = torch.zeros((1,y_dim))
            costs = [float(c) for c in input('enter costs: ').split(',')]
            for i in range(y_dim):
                mys[0,i] = costs[i]
            Y = torch.cat((Y, mys))
            record_dicts = record_dicts + [[]]

        manually_remove = input("do you want to remove last X? (y/[n]) : ")
        if manually_remove == 'y':
            X = X[:-1,:]

    print(X, X.shape) 
    print(Y, Y.size())
    print(X.shape, Y.size())
    input("review X and Y above")

    if resume:
        id_ = int(input("what was the last iter (1-indexed)? : "))
    else:
        id_ =0 
    
    first_run = move_gripper_first

    while id_ < num_iter:
        # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
        # appropriately adjust the punchout radius if the domain is normalized.

        gp_models = [get_and_fit_gp(X.float(),  Y[:, i : i + 1].reshape(-1,1)) for i in range(Y.shape[-1])]
        # model_list_gp = ModelListGP(gp_models[0],gp_models[1],gp_models[2])
        model_list_gp = ModelListGP(*gp_models)

        eci = ExpectedCoverageImprovement(
            model=model_list_gp,
            constraints=constraints,
            punchout_radius=punchout_radius,
            bounds=bounds,
            train_inputs=X.float(),
            num_samples=128,
            lambda_=0.0
        )
        next_X, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10 ,
            raw_samples=512,
        )
        next_X = next_X[0]

        print(f"\n\nIteration: {id_+1}/{num_iter},\n -> Datapoint selected: {next_X}\n")
        skip_str = input("press enter to proceed :) (or enter 'skip' without quotes to skip experiment)")
        skip_iter = (skip_str == 'skip')

        X = torch.cat((X, next_X.reshape(1,-1)))
        data = {'X':X, 'y_data':Y, 'record_dicts': record_dicts}

        with open(f"seed_{seed}_{lambda_}{run_suffix}.pkl", "wb") as f:
            pickle.dump(data,f)

        # next_y1,next_y2,next_y3,record_dict= system_agent(next_X.squeeze(),dim,seed,dataset_path,ckpt_path,skip_iter)
        # next_y = torch.tensor([next_y1,next_y2,next_y3]).reshape(1,3)
        next_ys,record_dict= system_agent(next_X.squeeze(),dim,y_dim,seed,dataset_path,ckpt_path,skip_iter,first_run)
        first_run = False

        Y = torch.cat((Y, next_ys))
        record_dicts.append(record_dict)
        id_+=1
        
        data = {'X':X, 'y_data':Y, 'record_dicts': record_dicts}

        print(data["X"])
        print(data["y_data"])
        print("Length of Y:",len(Y))

        with open(f"seed_{seed}_{lambda_}{run_suffix}.pkl", "wb") as f:
            pickle.dump(data,f)

    print("done with all iters")
    data = {'X':X, 'y_data':Y, 'record_dicts': record_dicts}
    print(data["X"])
    print(data["y_data"])
    print("Length of Y:",len(Y))

    input("press enter to continue ^-^ : ")

    with open(f"seed_{seed}_{lambda_}{run_suffix}.pkl", "wb") as f:
        pickle.dump(data,f)

    return X,Y#, model_list_gp

def get_initial(num_init,seed,dataset_path,ckpt_path,dim,y_dim,resume):
    data = {}

    if resume: 
        with open(f"GP_BO_init_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
        prev_Y = data["y_data"][1:,:]
        # print(prev_Y.size(dim=0))
        # input("asdf")
        prev_X = data["X"] #[:prev_Y.size(dim=0)+1,:]
        prev_record_dicts = data["record_dicts"]
    else:
        prev_X = np.empty(shape=(0,2))
        # prev_Y = torch.empty(size=(0,3))
        prev_Y = torch.empty(size=(0,y_dim))
        prev_record_dicts = []

    print(prev_X, prev_X.shape) 
    print(prev_Y, prev_Y.size())
    print(prev_X.shape, prev_Y.size())

    manually_append = input("do you want to manually append anything to prev? (y/n) : ")
    if manually_append == 'y':
        mys = torch.zeros((1,y_dim))
        costs = [float(c) for c in input('enter costs: ').split(',')]
        for i in range(y_dim):
            mys[0,i] = costs[i]
        prev_Y = torch.cat((prev_Y, mys))
        prev_X = data["X"]#[:prev_Y.size(dim=0),:]
        prev_record_dicts = prev_record_dicts + [[]]

    print(prev_X, prev_X.shape) 
    print(prev_Y, prev_Y.size())
    print(prev_X.shape, prev_Y.size())
    input("review prev_X and prev_Y above")
        
    X = np.empty((0,2))
    Y = torch.empty(size=(1,y_dim))
    record_dicts = []
    for n in range(prev_Y.size(dim=0),num_init):
        
        x = np.random.rand(1,2)
        print(x, x.shape)
        print(X, X.shape)
        print(f"this is run {n+1}/{num_init}, x = {x}")
        X = np.vstack((X, x))

        data = {
            'X': np.vstack((prev_X, X)), 
            'y_data': torch.cat((prev_Y,Y)), 
            'record_dicts': prev_record_dicts+record_dicts
        }
        with open(f"GP_BO_init_{seed}.pkl", "wb") as f:
            pickle.dump(data, f)

        next_ys = torch.zeros(size=(1,y_dim))

        next_ys,record_dict= \
            system_agent(x.squeeze(),dim,y_dim,seed,dataset_path,ckpt_path)
        # next_y = torch.tensor([next_y1,next_y2,next_y3]).reshape(1,3)
        Y = torch.cat((Y, next_ys))   
        record_dicts.append(record_dict)

        # model_gpr, mll = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
        data = {
            'X': np.vstack((prev_X, X)), 
            'y_data': torch.cat((prev_Y,Y)), 
            'record_dicts': prev_record_dicts+record_dicts
        }
        # print(data)
        print(data["X"])
        print(data["y_data"])
        # exit()

        with open(f"GP_BO_init_{seed}.pkl", "wb") as f:
            pickle.dump(data, f)
    
if __name__ == "__main__":
    # seed_list = [3000,5000,10000,15000,20000,25000,30000,35000,40000,45000]
    acf = "ECI"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--num_iter", type=int, default=20)
    # parser.add_argument("--delta", type=float, default=0.05)
    # parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--resume",  action='store_true', default=False)
    parser.add_argument("--lambda", type=float, default=0.5)
    parser.add_argument("--run-suffix", type=str, default="_c2c3_only")

    parser.add_argument("--print-all-init", action='store_true', default=False)
    parser.add_argument("--move-gripper-first", action='store_true', default=False)

    args = vars(parser.parse_args())
    num_init = args['num_init']
    seed = args['seed']
    num_iter = args['num_iter']

    torch.manual_seed(seed)
    random.seed(seed)

    dataset_path0 = path.join(home_dir, 'models/data/pusht_bayesian_cpu.zarr')
    ckpt_path0 = path.join(home_dir, 'models/push_T_diffusion_model_anjali_cuda10.pt')

    mode='bo'
    if mode=='initial':
        get_initial(num_init,seed,dataset_path0,ckpt_path0,2,args["resume"])
    else:
        
        if args['resume']:
            with open(f"seed_{seed}_{args['lambda']}{args['run_suffix']}.pkl", "rb") as f:
                data = pickle.load(f)
            X = data['X']
            y = data['y_data']
            record_dicts = data['record_dicts']
        else:    
            #open the random data file
            # with open(f"GP_BO_init_{seed}.pkl", 'rb') as f:
            with open(f"GP_BO_init_3000.pkl", 'rb') as f:
                data_initial = pickle.load(f)

            if args['print_all_init']:
                for i in range(len(data_initial['X'])):
                    print(i+1, data_initial['X'][i])
                # print(data_initial['y_data'])
                input('press enter to continue')
            
            #Load initial data
            X = data_initial['X'][0:10:2]
            # y = data_initial['y_data'][0:10:2]
            y = data_initial['y_data'][0:10:2,1:] # c2c3exp

            # Convert to torch tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            record_dicts = data_initial['record_dicts'][0:10:2]

        # Set the device and data type
        if torch.cuda.is_available():
            tkwargs = {
                "device": torch.device("cpu"),
                "dtype": torch.double,
            }
        else:
            tkwargs = {
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "dtype": torch.double,
                }

        bounds = torch.tensor([[0, 0], [1, 1]], **tkwargs)
        lb, ub = bounds
        dim = len(lb)
        delta1 = 0.3
        delta2 = 0.2
        delta3 = 0.3
        # delta = args['delta']
        # constraints = [("gt", delta1),("gt", delta2),("gt", delta3)]

        # c2c3exp
        constraints = [("gt", delta2),("gt", delta3)]

        punchout_radius =0.05# args['radius']

        y_dim = 2 # c2c3exp
        
        #gp_models
        X, y_data = fail_bo( 
            num_iter,bounds,X,y,
            constraints,dataset_path0,ckpt_path0,dim,y_dim,punchout_radius, 
            record_dicts, args['resume'], args['lambda'],args['run_suffix'],args['move_gripper_first'])
    
