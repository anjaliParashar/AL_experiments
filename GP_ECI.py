import sys
import os
import os.path as path
import random
root_dir = path.abspath(path.join(__file__ ,"../../.."))
home_dir = path.abspath(path.join(__file__ ,"../../../../.."))
sys.path.append(path.abspath(path.join(__file__ ,"../../..")))
sys.path.append(path.abspath(path.join(__file__ ,"../../../..")))
from utils import ExpectedCoverageImprovement, get_and_fit_gp, identify_samples_which_satisfy_constraints
import torch
from botorch.optim import optimize_acqf
from botorch.models import ModelListGP
import argparse
import pickle
import numpy as np
"""
Implements system agent. This is used as a baseline for ground truth, i.e., the best-case performance. 
The true reward is provided. 
"""
def unnormalize(X):
    return np.array([X[0]*350+150,X[1]*450]) # Unnormalize [0,1]X[0,1] to [100,400]X[100,400]


def get_user_feedback(X,dim,seed,dataset_path,ckpt_path):
    theta0=45
    score_human = run_diffusion_policy(x0=x0,y0=y0,theta0=np.deg2rad(theta0),seed=seed,dataset_path=dataset_path,ckpt_path=ckpt_path)
    return score_human

def system_agent(X,dim,seed,dataset_path,ckpt_path):
    score = get_user_feedback(X,dim,seed,dataset_path,ckpt_path)
    # if score<0.7:
    #     return torch.tensor(1)
    # else:
    #     return torch.tensor(1-score)
    return torch.tensor(score)

def fail_bo(num_iter,bounds,X,Y,constraints,dataset_path,ckpt_path,dim,punchout_radius=0.1):
    id_ =0 
    while id_ < num_iter:
        # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
        # appropriately adjust the punchout radius if the domain is normalized.
        gp_models = [get_and_fit_gp(X.float(),  Y[:, i : i + 1].reshape(-1,1)) for i in range(Y.shape[-1])]
        model_list_gp = ModelListGP(gp_models[0],gp_models[1],gp_models[2])
        eci = ExpectedCoverageImprovement(
            model=model_list_gp,
            constraints=constraints,
            punchout_radius=punchout_radius,
            bounds=bounds,
            train_inputs=X.float(),
            num_samples=128,
        )
        next_X, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10 ,
            raw_samples=512,
        )
        next_X = next_X[0]
        print("Iteration:",id_,"Datapoint selected:",next_X)

        next_y1,next_y2,next_y3= system_agent(next_X.squeeze(),dim,seed,dataset_path,ckpt_path)
        next_y = torch.tensor([next_y1,next_y2,next_y3]).reshape(1,2)
        X = torch.cat((X, next_X.reshape(1,-1)))
        Y = torch.cat((Y, next_y))
        
        id_+=1
        data = {'X':X, 'y_data':Y}
        print("Length of Y:",len(Y))
        file = open(f"/experiment_logs_ECI/seed_{seed}.pkl", "wb")
        pickle.dump(data,file )
        file.close()
    return X,y_data, gp_models

def get_initial(num_init,seed,dataset_path,ckpt_path,dim):
    X = np.random.rand(num_init,dim)
    Y = torch.zeros((1,3))
    for x in X:
        next_y1,next_y2,next_y3= system_agent(x.squeeze(),dim,seed,dataset_path,ckpt_path)
        next_y = torch.tensor([next_y1,next_y2,next_y3]).reshape(1,3)
        Y = torch.cat((Y, next_y))   
        # model_gpr, mll = init_and_fit_gpr(torch.tensor(X_collect).double(), torch.tensor(np.array(y_data)).double().reshape(-1,1))
        data = {'X':X, 'y_data':Y[1:,:]}
        file = open(f"GP_BO_init_{seed}.pkl", "wb")
        pickle.dump(data,file )
        file.close()
    
if __name__ == "__main__":
    # seed_list = [3000,5000,10000,15000,20000,25000,30000,35000,40000,45000]
    acf = "ECI"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_init", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_iter", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--radius", type=float)

    args = vars(parser.parse_args())
    num_init = args['num_init']
    seed = args['seed']
    num_iter = args['num_iter']

    torch.manual_seed(seed)
    random.seed(seed)

    dataset_path0 = path.join(home_dir, 'models/data/pusht_bayesian_cpu.zarr')
    ckpt_path0 = path.join(home_dir, 'models/push_T_diffusion_model_anjali_cuda10.pt')

    mode='initial'
    if mode=='initial':
        get_initial(num_init,seed,dataset_path0,ckpt_path0,dim=2)
    else:
        #open the random data file
        with open(f"GP_BO_init_{3000}.pkl", 'rb') as f:
            data_initial = pickle.load(f)
        
        #Load initial data
        X = data_initial['X']
        y = data_initial['y']
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

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
        delta = args['delta']
        constraints = [("lt", delta)]
        punchout_radius = args['radius']
        
        
        X,y_data, gp_models = fail_bo(num_iter,bounds,X,y,constraints,dataset_path0,ckpt_path0,dim,punchout_radius)
    
    