import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import matplotlib
matplotlib.use("Agg")
#warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
# from Simulator.RBFleX import RBFleX
from Simulator.Computation import DL2
# from Simulator.scale_sim.scale import scale as ScaleSim
# from Simulator.defines.single_run import mainstage
from pymoo.algorithms.moo.nsga2 import NSGA2
import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
# Multi Objective Bayesian Optimization
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize, standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning,)
from collections import defaultdict
import time
import shutil
import sys
import pandas as pd
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine
from pyswarm import pso
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from scipy.stats import norm
import random
import os

from typing import Dict, List, Any, Tuple

import warnings
from pathlib import Path

from pprint import pprint
import re

import matplotlib.patheffects as pe

import gc
from contextlib import suppress

from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message="A not p.d., added jitter")

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.double
else:
    DEVICE = "cpu"
    DTYPE = torch.double
tkwargs = {
    "dtype": DTYPE,
    "device": DEVICE,
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

clock_frequency_hz = 2.2 * 1e9

hvs = []

# SWoptOnlyrandom_ppl: fixed HW tail (last 6 dims) sampled once and reused globally.
GLOBAL_SW_OPT_ONLY_FIXED_HW = None

class FRCN_Simulator:
    def __init__(self, 
                IN_H, 
                IN_W,
                opt_mode,
                set_hd_bounds,
                set_fs,  
                optimized_components, 
                dataset="ImageNet", # cifar100, ImageNet, cifar10
                score_input_batch=8, 
                img_root="xxx",
                benchmark="xxx", # sss, lora, qwen-lora
                benchmark_root="D:/COFleX/COFleX/NATS-sss-v1_0-50262-simple",
                # benchmark_root="D:/COFleX/COFleX/NATS-tss-v1_0-3ffb9-simple", # NATS-Bench/NATS-tss-v1_0-3ffb9-simple
                n_hyper=0, 
                ref_score=-0,
                iters=0, 
                mc_samples=128, 
                acqu_algo=" ",

                seed=42,

                model_id = " ",

                batch_size=0, 
                num_restarts=10, 
                raw_samples=512, 
                n_init_size=0, 
                mapping=" ",
                hw='Meta_prototype_DF',
                Hardware_Arch=" ",
                TOTAL_RUN_TIME = 0,
        ):
        self.hw = hw
        self.set_opt_mode = opt_mode
        self.set_hd_bounds = set_hd_bounds
        self.set_fs = [int(float(x)) for x in set_fs.split()]
        self.set_acqu_algo = acqu_algo #
        self.set_iters = iters #
        self.set_n = n_init_size #
        self.set_Hardware_Arch = Hardware_Arch #
        self.set_mapping = mapping #
        self.current_iteration = 0
        self.dataset = dataset
        self.extra_rd_flag = False
        self.cluster_lim = 2
        # ------------------- Variables ----------------------- # 
        self.observation_theta = []
        self.observation_gamma = []
        self.beta = []
        self.IntPar = []
        self.DIntPar = []
        self.status = "NOM"

        self.err_list = []
        self.arch_list = []
        self.ranked_score_list = []
        
        self.search_time_rec = None
        self.start_time = 0
        self.cost_time = 0
        self.estm_vist_time = 0
        self.GPU_H = 0
        self.TOTAL_RUN_TIME = TOTAL_RUN_TIME
        
        self.TOTAL_TRAIN_OBJ_REC = None
        self.TOTAL_TRAIN_X_REC = None

        self.iter = 0
        self.OPT_VS_TIME_REC = None

        self.LEVELS = {
            0: np.array(list(range(12))),                                     # sw_pos (6 single + 6 combo)
            1: np.array([0,1,2,4,8,16,32,64,128,256,512]),                    # sw_rank
            2: np.array([2]),                                                 # sw_prec (固定)
            3: np.array(list(range(45))),                                     # sw_layer (0~35 single + 36~44 combo)
            4: np.array([2]),                                                 # hw_prec (固定)
            5: np.array([2,4,6,8]),                                           # nmp_channel_num
            6: np.array([4,8,16,32,64,128,256,512,1024]),                     # i_buff
            7: np.array([4,8,16,32,64,128,256,512,1024]),                     # w_buff
            8: np.array([4,8,16,32,64,128,256,512,1024]),                     # o_buff
            9: np.array(list(range(16))),             # bw_space
        }

        self.LEVELS_sw = {
            0: np.array(list(range(12))),                                     # sw_pos
            1: np.array([0,1,2,4,8,16,32,64,128,256,512]),                    # sw_rank
            2: np.array([2]),                                                 # sw_prec (固定)
            3: np.array(list(range(45))),                                     # sw_layer
        }

        self.LEVELS_hw = {
            0: np.array([2]),                                                 # hw_prec (固定)
            1: np.array([2,4,6,8]),                                           # nmp_channel_num
            2: np.array([4,8,16,32,64,128,256,512,1024]),                     # i_buff
            3: np.array([4,8,16,32,64,128,256,512,1024]),                     # w_buff
            4: np.array([4,8,16,32,64,128,256,512,1024]),                     # o_buff
            5: np.array(list(range(16))),                                     # bw_space
        }

        # "self_attn.q_proj",
        # "self_attn.v_proj",
        # "self_attn.o_proj",
        # "mlp.gate_proj",
        # "mlp.up_proj",
        # "mlp.down_proj",
        # "self_attn.qv_pair" 

        self.pos2id = {
        "q_proj": 0,
        "k_proj": 1,
        "v_proj": 2,
        "out_proj": 3,
        "fc1": 4,
        "fc2": 5,
        "__combo_attn_all__": 6,
        "__combo_attn_core__": 7,
        "__combo_kv_pair__": 8,
        "__combo_mlp_only__": 9,
        "__combo_out_fc2__": 10,
        "__combo_full_mix__": 11,
        }

        # PRECISIONS = ["int8", "int4", "fp16"]
        self.prec2id = {
        "int8": 0,
        "int4": 1,
        "fp16": 2,
        }

        self.total_candidates=0
        self.total_oracle=0 
        self.global_oracle_ratio=0
        
        self.hist_X = None      
        self.hist_Y = None      
        self.arc_X  = None      
        self.arc_Y  = None
        self.EX     = None      

        self.is_score_based = True

        self.new_add_Y_err = None 
        self.new_add_Y_eng = None

        self.nsga_archive = {}
        self.cache_hits = 0
        self.cache_miss = 0

        if self.set_acqu_algo in ["Coflex_ppl","qNParEGO_ppl","qEHVI_ppl","qNEHVI_ppl","random_ppl","nsga_ppl","SWoptOnlyrandom_ppl","HWoptOnlyrandom_ppl"]:
            print("\033[1;31m[ --------------------------------------------------- Start PPL Based HW-NAS --------------------------------------------------- ]\033[0m")
            self.is_score_based = False
        else:
            print("\033[1;31m[ --------------------------------------------------- Start Score Based HW-NAS --------------------------------------------------- ]\033[0m")
            self.is_score_based = True

        self.seed = seed
        self.model_id = model_id

        self.time_used_list = []

        global GLOBAL_SW_OPT_ONLY_FIXED_HW
        if self.set_acqu_algo == "SWoptOnlyrandom_ppl" and GLOBAL_SW_OPT_ONLY_FIXED_HW is None:
            GLOBAL_SW_OPT_ONLY_FIXED_HW = np.array(
                [np.random.choice(self.LEVELS_hw[i]) for i in range(6)],
                dtype=int,
            )
            print(f"[SWoptOnlyrandom_ppl] Fixed HW tail (global): {GLOBAL_SW_OPT_ONLY_FIXED_HW.tolist()}")

        self.ts = None
        self.need_eval_model_type=None

        self.proxy_model = None
        self.proxy_std = None

        # ------------------- Variables ----------------------- # 
        if self.extra_rd_flag == True:
            self.set_n = 1
        if self.dataset == "ImageNet":
            img_root="D:/OneDrive - Singapore University of Technology and Design/COFleX_imagenet/COFleX/dataset/ImageNet"
        if self.set_acqu_algo == 'nsga':
            self.set_iters = 1
            
        print("___Configutations___", 
        self.set_acqu_algo, 
        self.set_opt_mode, 
        self.set_hd_bounds[0], 
        self.set_hd_bounds[1], 
        self.set_hd_bounds[2], 
        self.set_hd_bounds[3], 
        self.set_fs[0], 
        self.set_fs[1], 
        self.set_fs[2], 
        self.set_fs[3], 
        self.set_fs[4], 
        self.set_iters, 
        self.set_n, 
        self.set_Hardware_Arch, 
        self.set_mapping,
        self.extra_rd_flag,
        self.hw,
        self.dataset,
        img_root,
        )
        print('+++++++++++ Configuration +++++++++++')

        #######################################
        # General config    
        #######################################
        print("[General config]")
        self.n_init_size = n_init_size
        self.image = torch.ones(1,3,IN_H, IN_W)
        print("\tInput resolution: [{}, {}]".format(IN_H, IN_W))
        print("\tAcquation function: {}".format(acqu_algo))
        print("\tBatch size for initial data generation: {}".format(n_init_size))
        print("\tDevice: {}".format(DEVICE))
        #######################################
        # Config for NAS     
        #######################################
        print("[RBFleX config]")
        self.batch_size_score = n_hyper # The number of batch images for hyperparameter detection algorithm
        self.benchmark = benchmark
        print("\tbatch images for HDA: {}".format(self.batch_size_score))
        #######################################
        # Config for DSE    
        #######################################
        print("[DSE config]")
        self.optimized_comp = optimized_components
        self.opt_architecture = 0
        self.hardware_components_values = {"X1": 0, "X2": 0, "X3": 0, "X4": 0, "X5": 0}

        self.opt_params = [k for k, v in self.optimized_comp.items() if v == 0]
        self.not_opt_params = [k for k, v in self.optimized_comp.items() if v != 0]
        
        for key in self.not_opt_params:
            self.hardware_components_values[key] = (self.optimized_comp[key],self.optimized_comp[key])
        for key in self.opt_params:
            self.hardware_components_values[key] = (20,36)
        
        self.Hardware_Arch = Hardware_Arch
        if self.Hardware_Arch == "DL2":
            self.Num_HWopt = 6

        elif self.Hardware_Arch == "ScaleSim":
            self.Num_HWopt = 6
        
        elif self.Hardware_Arch == "DeFiNES":
            self.Num_HWopt = 6  # Number of HW varies to be optimized
            self.Num_SWopt = 5
        # H^2
        elif self.Hardware_Arch == "H^2":
            self.Num_HWopt = 5  # Number of HW varies to be optimized
            self.Num_SWopt = 4  # Number of SW varies to be optimized
        
        self.bounds_eng = [[0] * self.Num_HWopt, [0] * self.Num_HWopt]
        self.bounds_err = [[0] * self.Num_SWopt, [0] * self.Num_SWopt]
        
        print("Hardware Architecture: {}".format(self.Hardware_Arch))
        
        print('\tTo be optimized: {}'.format(self.opt_params))
        print('\tFixed HW params: {}'.format(self.not_opt_params))
        
        if self.Hardware_Arch == "DL2":
            if not mapping in ["rs", "ws"]:
                print("[ERROR] mapping for DL2 supports only [rs, ws].")

        elif self.Hardware_Arch == "ScaleSim":
            if not mapping in ["os", "ws", "is"]:
                print("[ERROR] mapping for systolic array supports only [os, ws, is].")
        
        elif self.Hardware_Arch == "DeFiNES":
            if not mapping in ["os", "ws", "is"]:
                print("[ERROR] mapping for systolic array supports only [os, ws, is].")
        
        elif self.Hardware_Arch == "H^2":
            if not mapping in ["os", "ws", "is"]:
                print("[ERROR] mapping for systolic array supports only [os, ws, is].")
        
        print('\tMapping: {} stationary'.format(mapping))

        #######################################
        #  Config for Multiple object baysian optimazation
        #######################################
        # NAS
        if self.benchmark == "sss":
            self.nas_dim = 5
            self.nas_obj = 1
            self.sf_lower_bound = 8
            self.sf_upper_bound = 64
            self.sf_bounds = torch.stack([torch.ones(self.nas_dim, **tkwargs), 8.0*torch.ones(self.nas_dim, **tkwargs)])
            self.sf_norm = torch.stack([self.sf_lower_bound*torch.ones(self.nas_dim, **tkwargs), self.sf_upper_bound*torch.ones(self.nas_dim, **tkwargs)])
            self.sf_standard_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])

        elif self.benchmark == "lora" or self.benchmark == "qwen-lora":
            self.nas_dim = 4
            self.nas_obj = 1
            
            lb = torch.tensor([0.0,0.0,0.0,0.0], **tkwargs)
            ub = torch.tensor([6.0,512.0,2.0,15.0], **tkwargs)

            self.sf_bounds = torch.stack([lb, ub], dim=0)
            self.sf_norm = torch.stack([lb, ub], dim=0)

            self.sf_standard_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])

            # print("\033[92m[ok]\033[0m")
            # pprint({
            #     "sf_bounds": self.sf_bounds,
            #     "sf_norm": self.sf_norm,
            #     "sf_standard_bounds": self.sf_standard_bounds,
            # })

        elif self.benchmark == "201":
            from design_space.models import get_search_spaces
            from design_space.policy import PolicyTopology
            self.space_201 = get_search_spaces('cell', 'nas-bench-201')
            self.policy = PolicyTopology(self.space_201 )
            self.nas_dim = 6
            self.nas_obj = 1
            self.sf_lower_bound = 0
            self.sf_upper_bound = 4
            self.sf_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), 4.0*torch.ones(self.nas_dim, **tkwargs)])
            self.sf_norm  = self.sf_bounds
            self.sf_standard_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])
        
        self.hd_dim = len(self.opt_params)
        if self.Hardware_Arch == "DL2":
            self.hd_obj = 2 #[energy and cycle]
            self.SCORE_IDX = 0
            self.ENERGY_IDX = 1
            self.CYCLE_IDX = 2

        elif self.Hardware_Arch == "ScaleSim":
            self.hd_obj = 1 #[cycle]
            self.SCORE_IDX = 0
            self.CYCLE_IDX = 1

        elif self.Hardware_Arch == "DeFiNES":
            self.hd_obj = 1  # [edp]
            self.ERROR_IDX = 0
            # self.ENERGY_IDX = 1
            # self.CYCLE_IDX = 2
            self.EDP_IDX = 1
        
        elif self.Hardware_Arch == "H^2":
            self.hd_obj = 1  # [lat]
            self.ERROR_IDX = 0
            # self.ENERGY_IDX = 1
            # self.CYCLE_IDX = 2
            self.LAT_IDX = 1
        
        self.mobo_dim = self.nas_dim + self.hd_dim # how many obj to be optimized
        self.mobo_obj = self.nas_obj + self.hd_obj # how many output

        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "mobo_dim": self.mobo_dim,
        #     "mobo_obj": self.mobo_obj,
        # })

        self.ref_point = 15*torch.ones(self.mobo_obj, **tkwargs)  # reference point
        
        if self.Hardware_Arch == "DL2":
            self.hd_lower_bound = [20,20]    #[0]: for PE array [1]: for memory
            self.hd_upper_bound = [60,60]    #[0]: for PE array [1]: for memory

        elif self.Hardware_Arch == "ScaleSim":
            self.hd_lower_bound = [1,10]     #[0]: for PE array [1]: for memory
            self.hd_upper_bound = [64,512]   #[0]: for PE array [1]: for memory
        
        elif self.Hardware_Arch == "DeFiNES":
            self.hd_lower_bound = [int(self.set_hd_bounds[0]), int(self.set_hd_bounds[2])]  # [0]: for PE array [1]: for memory
            self.hd_upper_bound = [int(self.set_hd_bounds[1]), int(self.set_hd_bounds[3])]  # [0]: for PE array [1]: for memory

        elif self.Hardware_Arch == "H^2":
            self.hd_lower_bound = [int(self.set_hd_bounds[0]), int(self.set_hd_bounds[2]), int(self.set_hd_bounds[4])]  # [0]: for PE array [1]: for memory
            self.hd_upper_bound = [int(self.set_hd_bounds[1]), int(self.set_hd_bounds[3]), int(self.set_hd_bounds[5])]  # [0]: for PE array [1]: for memory
        
        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "hd_lower_bound": self.hd_lower_bound,
        #     "hd_upper_bound": self.hd_upper_bound,
        # })
        
        # ---
        nk = 0
        self.hd_bounds = [[0]*self.Num_HWopt,[0]*self.Num_HWopt]
        for k, v in self.optimized_comp.items():
            if v == 0:
                if k == "X1":
                    self.hd_bounds[0][nk] = self.hd_lower_bound[0]
                    self.hd_bounds[1][nk] = self.hd_upper_bound[0]
                    # self.bounds_eng[0][nk] = self.hd_lower_bound[0]
                    # self.bounds_eng[1][nk] = self.hd_upper_bound[0]  
                elif k == "X2" or k == "X3" or k == "X4":
                    self.hd_bounds[0][nk] = self.hd_lower_bound[1]
                    self.hd_bounds[1][nk] = self.hd_upper_bound[1]
                    # self.bounds_eng[0][nk] = self.hd_lower_bound[1]
                    # self.bounds_eng[1][nk] = self.hd_upper_bound[1]
                elif k == "X5":
                    self.hd_bounds[0][nk] = self.hd_lower_bound[2]
                    self.hd_bounds[1][nk] = self.hd_upper_bound[2]
                    # self.bounds_eng[0][nk] = self.hd_lower_bound[2]
                    # self.bounds_eng[1][nk] = self.hd_upper_bound[2]
            else:
                self.hd_bounds[0][nk] = v
                self.hd_bounds[1][nk] = v
                self.bounds_eng[0][nk] = v
                self.bounds_eng[1][nk] = v
            nk += 1
        self.hd_bounds = torch.tensor(self.hd_bounds, **tkwargs)
        
        # ---
        nk = 0
        self.hd_standard_bounds = torch.zeros(2, self.hd_dim, **tkwargs)
        self.hd_standard_bounds[1] = 1
        for k, v in self.optimized_comp.items():
            if not v == 0:
                self.hd_standard_bounds[0][nk] = 1
            nk += 1

        # ---
        nk = 0
        self.hd_norm = [[0]*self.Num_HWopt,[0]*self.Num_HWopt]
        for k, v in self.optimized_comp.items():
            if v == 0:
                if k == "X1":
                    self.hd_norm[0][nk] = self.hd_lower_bound[0]
                    self.hd_norm[1][nk] = self.hd_upper_bound[0]
                elif k == "X2" or k == "X3" or k == "X4":
                    self.hd_norm[0][nk] = self.hd_lower_bound[1]
                    self.hd_norm[1][nk] = self.hd_upper_bound[1]
                elif k == "X5":
                    self.hd_norm[0][nk] = self.hd_lower_bound[2]
                    self.hd_norm[1][nk] = self.hd_upper_bound[2]
            else:
                self.hd_norm[0][nk] = 0
                self.hd_norm[1][nk] = v
            nk += 1
        self.hd_norm = torch.tensor(self.hd_norm, **tkwargs)

        sf_last_col = self.sf_bounds[:, -2:-1].to(self.hd_bounds.dtype)
        sf_last_col_norm = self.sf_norm[:, -2:-1].to(self.hd_norm.dtype)
        sf_last_col_std = self.sf_standard_bounds[:, -2:-1].to(self.hd_standard_bounds.dtype)

        self.hd_bounds = torch.cat((sf_last_col, self.hd_bounds), dim=1)
        self.hd_norm = torch.cat((sf_last_col_norm, self.hd_norm), dim=1)
        self.hd_standard_bounds = torch.cat((sf_last_col_std, self.hd_standard_bounds), dim=1)

        self.bounds = torch.cat((self.sf_bounds, self.hd_bounds),1)
        self.bounds_fornorm = torch.cat((self.sf_norm,self.hd_norm),1)
        self.bounds_forstard = torch.cat((self.sf_standard_bounds, self.hd_standard_bounds),1)

        # ---- pretty bounds printer ---------------------------------------------------
        GREEN = "\033[92m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"

        def _to_1d(t):
            try:
                t = t.detach().cpu().float()
            except Exception:
                pass
            return t

        def _fmt(x):
            x = float(x)
            return str(int(x)) if x.is_integer() else f"{x:.4g}"

        def print_bounds_table(name: str, bounds, sf_dim: int):
            b = _to_1d(bounds)
            lower, upper = b[0].tolist(), b[1].tolist()
            D = len(lower)

            titles = ['pos', 'rank', 'prec_sw', 'layer', 'prec_hw', 'nmp_c', 'i_buff', 'o_buff', 'w_buf', 'bw']

            widths = [max(len(titles[i]), len(_fmt(lower[i])), len(_fmt(upper[i])), 3) for i in range(D)]

            def row(label, arr):
                pieces = [f"{label:<10}| "]
                for i, x in enumerate(arr):
                    pieces.append(f"{_fmt(x):>{widths[i]}} | ")
                    if i == sf_dim-1:  
                        pieces.append(f"{GREEN}{BOLD}||{RESET} ")
                return "".join(pieces).rstrip()

            total_w = 12 + sum(w+3 for w in widths) + 3  
            horiz = "-" * total_w

            header_cells = [f"{'name':<10}| "]
            for i, t in enumerate(titles):
                header_cells.append(f"{t:>{widths[i]}} | ")
                if i == sf_dim-1:
                    header_cells.append(f"{GREEN}{BOLD}||{RESET} ")
            header = "".join(header_cells).rstrip()

            tag_cells = [f"{'':<10}| "]
            for i in range(D):
                tag = "sf" if i < sf_dim else "hd"
                tag_cells.append(f"{tag:>{widths[i]}} | ")
                if i == sf_dim-1:
                    tag_cells.append(f"{GREEN}{BOLD}||{RESET} ")
            tagline = "".join(tag_cells).rstrip()

            print(f"{BOLD}{name}{RESET}")
            print(horiz)
            print(header)
            print(horiz)
            print(tagline)
            print(horiz)
            print(row("lower", lower))
            print(horiz)
            print(row("upper", upper))
            print(horiz)
        # ------------------------------------------------------------------------------
        
        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "bounds": self.bounds,
        #     "bounds_fornorm": self.bounds_fornorm,
        #     "bounds_forstard": self.bounds_forstard,
        # })

        for nm in ("bounds", "bounds_fornorm", "bounds_forstard"):
            print_bounds_table(nm, getattr(self, nm), sf_dim=self.Num_SWopt)

        
        self.NUM_RESTARTS = num_restarts if not SMOKE_TEST else 2
        self.RAW_SAMPLES = raw_samples if not SMOKE_TEST else 4
        self.N_BATCH = iters if not SMOKE_TEST else 10 # number of iteration
        self.MC_SAMPLES = mc_samples if not SMOKE_TEST else 16
        self.acqu_algo = acqu_algo
        
        if self.acqu_algo == "nsga" or self.acqu_algo == "nsga_ppl":
            self.BATCH_SIZE = 8
        else:
            self.BATCH_SIZE = batch_size

        self.bounds_eng = self.bounds[:, 4:]
        self.bounds_err = self.bounds[:, :4]
        
        self.bounds_forstard_eng = self.bounds_forstard[:, 4:]
        self.bounds_forstard_err = self.bounds_forstard[:, :4]

        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "bounds_eng": self.bounds_eng,
        #     "bounds_err": self.bounds_err,
        #     "bounds_forstard_eng": self.bounds_forstard_eng,
        #     "bounds_forstard_err": self.bounds_forstard_err,
        # })
        
        #######################################
        # Initilize RBFleX, DSE, and Estimator    
        #######################################
        if self.Hardware_Arch in ("DL2", "ScaleSim", "DeFiNES"): 
            self.RBFleX = RBFleX(dataset, score_input_batch, img_root, benchmark, benchmark_root, self.batch_size_score)
            if self.Hardware_Arch == "DL2":
                self.AutoDSE = DL2(self.optimized_comp, mapping)
            elif self.Hardware_Arch == "ScaleSim":
                self.AutoDSE = ScaleSim(mapping)
            elif self.Hardware_Arch == "DeFiNES":
                self.AutoDSE = None
            self.mapping = mapping

    def snap_to_levels_batch(self, X: torch.Tensor, levels_map: dict, cols=None) -> torch.Tensor:
        """
        X: (B, D) torch.Tensor，原始数值域（非 0-1）
        levels_map: {col_idx: np.ndarray([...])}，每列允许的离散集合（原始域）
        cols: 需要贴网格的列索引集合；默认按 levels_map 的 key
        """
        Xr = X.clone()
        if Xr.ndim == 1:
            Xr = Xr.unsqueeze(0)  # 统一 (B,D)
        use_cols = cols
        for j in use_cols:
            lv_np = levels_map[j]
            lv = torch.as_tensor(lv_np, dtype=Xr.dtype, device=Xr.device)  # (L,)
            # (B,1) - (1,L) → (B,L)
            dist = (Xr[:, j].unsqueeze(1) - lv.unsqueeze(0)).abs()
            idx  = dist.argmin(dim=1)           # (B,)
            Xr[:, j] = lv[idx]                  # 贴最近的合法值
        return Xr

    def delete_all_folders(self, path):
        if not os.path.exists(path):
            print(f"The path {path} does not exist.")
            return
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                except FileNotFoundError as e:
                    pass
            else:
                try:
                    os.remove(item_path)
                except FileNotFoundError as e:
                    pass
   
    def print_mem(self, tag=""):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()/1024**2
            reserv = torch.cuda.memory_reserved()/1024**2
            print(f"[{tag}] allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

    def cleanup_proxy_ckpt(self, ckpt_path="proxy_ckpt.pt"):
        if not self.is_score_based:
            return
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"[cleanup] Error!! Foud exit proxy checkpoint, remove and try again: {ckpt_path}")
                exit()
        except Exception as exc:
            print(f"[cleanup] failed to remove proxy checkpoint {ckpt_path}: {exc}")
    
    def hard_free(self, model=None, tokenizer=None, *others):
        # # 0) 先把还活着的 tensor 迁回 CPU，避免 GPU 上有梯度图/缓存
        # with suppress(Exception):
        #     if model is not None: model.to("cpu")

        # 1) 断引用（包括 optimizer / scheduler / dataloader / outputs / loss 等）
        for obj in (tokenizer, model, *others):
            with suppress(Exception):
                del obj

        # 2) Python GC
        gc.collect()

        # 3) 等所有 CUDA kernel 结束，再清缓存
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()          # 跨进程句柄
            time.sleep(0.1)                    # 给驱动一点时间（兜底）

        self.print_mem("after hard_free")

    def _rbflex_and_dse(self, image, candi_network, candi_hardparams):
        self.estm_vist_time += 1
        c_list = candi_network.split(':')

        # build searchspace--->.json
        HERE = Path(__file__).resolve()
        ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
        sys.path.insert(0, str(ROOT_LORA))

        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
        make_searchspace(c_list[0], c_list[1], c_list[2], c_list[3], self.seed)
                        #  pos,        r,        prec,        layer=0,        sd=42

        #######################
        #         DSE         
        #######################
        ROOT_DSE= Path(__file__).resolve().parents[2]   # …/H2-LLM-ISCA-2025
        if str(ROOT_DSE) not in sys.path:
            sys.path.insert(0, str(ROOT_DSE))

        from main import make_parse_or_dse

        assert self.model_id is not None, "Check model_id!!"

        need_eval_model_id = self.model_id.split("/")[1]

        # Llama-3.2-3B / Qwen2.5-1.5B / Llama-3.1-8B
        if need_eval_model_id == 'Llama-3.1-8B': 
            self.need_eval_model_type = 'llama-lora'

        elif need_eval_model_id == 'Qwen2.5-1.5B': 
            self.need_eval_model_type = 'qwen-lora'
        
        elif need_eval_model_id == 'Llama-3.2-3B': 
            self.need_eval_model_type = 'llama3.2-3b-lora'          
        elif need_eval_model_id == 'esm2_t36_3B_UR50D':
            self.need_eval_model_type = 'esm2-3b-lora'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. 转为 list[int]
        net_list = [int(x) for x in candi_network.split(":")]   # [3, 7, 2, 9]

        # 3. 变成 2D Tensor，带 tkwargs（device, dtype）
        net_tensor  = torch.tensor([net_list], **tkwargs)          # shape: (1, 4)
        hard_tensor = torch.tensor([candi_hardparams], **tkwargs)  # shape: (1, 6)

        # 4. 在 Tensor 上 snap 到合法档位
        #    注意 cols 范围要覆盖到所有列：range(4) 和 range(6)
        net_tensor  = self.snap_to_levels_batch(net_tensor,  levels_map=self.LEVELS_sw, cols=range(4))
        hard_tensor = self.snap_to_levels_batch(hard_tensor, levels_map=self.LEVELS_hw, cols=range(6))

        # 5. 把 snapped Tensor 再转回 Python 类型
        #    这里建议先 detach().cpu() 再转 int 和 list
        net_list_snapped = (
            net_tensor
            .detach()
            .cpu()
            .squeeze(0)        # (1,4) -> (4,)
            .to(torch.int64)
            .tolist()
        )
        hardparams_snapped = (
            hard_tensor
            .detach()
            .cpu()
            .squeeze(0)        # (1,6) -> (6,)
            .to(torch.int64)
            .tolist()
        )

        # 6. 还原成你原来的两种变量形式
        candi_network    = ":".join(str(v) for v in net_list_snapped)   # "a:b:c:d"
        candi_hardparams = hardparams_snapped                           # list[int]

        print("\033[92m[ok]\033[0m")
        pprint({
            "candi_network": candi_network,
            "candi_hardparams": candi_hardparams,
        })

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # candi_network, candi_hardparams
        make_parse_or_dse('parse', candi_hardparams[0], candi_hardparams[1], candi_hardparams[2], candi_hardparams[3], candi_hardparams[4], candi_hardparams[5], self.need_eval_model_type)
        make_parse_or_dse('dse', candi_hardparams[0], candi_hardparams[1], candi_hardparams[2], candi_hardparams[3], candi_hardparams[4], candi_hardparams[5], self.need_eval_model_type)

        def parse_result_log(log_dir: Path):
            # /home/myh/H2-LLM-ISCA-2025/hw_nas/kick_the_tires/llama3.2-3b-lora

            log = sorted((log_dir).rglob("result.log"), key=lambda p: p.stat().st_mtime)[-1]
            text = log.read_text(encoding="utf-8")

            def grab(pattern, cast=float, default=None):
                m = re.search(pattern, text)
                if not m: return default
                return cast(m.group(1))

            idv     = grab(r"Best idv latency is ([0-9.eE+-]+)")
            prefill = grab(r"Prefill latency is ([0-9.eE+-]+)")
            decode  = grab(r"Decoding latency is ([0-9.eE+-]+)")

            layer_num = grab(r"Total layer num is ([0-9.eE+-]+)")
            token_num = grab(r"Total tokens is ([0-9.eE+-]+)")
            ttft = grab(r"TTFT is ([0-9.eE+-]+)")
            tpot = grab(r"TPOT is ([0-9.eE+-]+)")
            end_to_end_latency = grab(r"End-to-end Latency is ([0-9.eE+-]+)")
            goodput = grab(r"Goodput is ([0-9.eE+-]+)")
            throughput = grab(r"Throughput is ([0-9.eE+-]+)")

            m = re.search(r"PE config\s+(\d+)\s+fpus\s*@\s*([0-9.]+)GHz,\s*hybrid bonding bandwidth\s*([0-9.]+)GB/s", text)
            fpu, freq, hb_bw = (int(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)

            m = re.search(r"Input buffer size\s*([0-9.]+)KB,\s*weight buffer size\s*([0-9.]+)KB,\s*output buffer size\s*([0-9.]+)KB,\s*NMP channel num\s*([0-9]+)", text)
            ib_kb, wb_kb, ob_kb, nmp = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else (None, None, None, None)

            return dict(idv=idv, prefill=prefill, decode=decode,
                        fpu=fpu, freq=freq, hb_bw=hb_bw,
                        ib_kb=ib_kb, wb_kb=wb_kb, ob_kb=ob_kb, nmp=nmp, 
                        layer_num=layer_num, token_num=token_num, 
                        ttft=ttft, tpot=tpot, end_to_end_latency=end_to_end_latency,
                        goodput=goodput,throughput=throughput,)

        def _mk_line(widths, ch='-'):
            return '+' + '+'.join(ch*(w+2) for w in widths) + '+'

        def _mk_row(widths, cols):
            s = []
            for w, c in zip(widths, cols):
                s.append(' ' + str(c).ljust(w) + ' ')
            return '|' + '|'.join(s) + '|'

        def print_tables(metrics: dict, title="LLM Inference Summary"):
            head1 = ["Metric", "Pre Layer Latency (s)"]
            rows1 = [
                ["Best idv", f"{metrics['idv']:.12f}" if metrics['idv'] is not None else "—"],
                ["Prefill", f"{metrics['prefill']:.12f}" if metrics['prefill'] is not None else "—"],
                ["Decoding", f"{metrics['decode']:.12f}" if metrics['decode'] is not None else "—"],
            ]
            widths1 = [max(len(h), max(len(str(r[i])) for r in rows1)) for i, h in enumerate(head1)]

            ib = int(metrics['ib_kb']) if metrics['ib_kb'] is not None else "—"
            wb = int(metrics['wb_kb']) if metrics['wb_kb'] is not None else "—"
            ob = int(metrics['ob_kb']) if metrics['ob_kb'] is not None else "—"
            nmp = metrics['nmp'] if metrics['nmp'] is not None else "—"
            fpu = metrics['fpu'] if metrics['fpu'] is not None else "—"
            freq= f"{metrics['freq']:.1f} GHz" if metrics['freq'] is not None else "—"
            hb  = f"{metrics['hb_bw']:.1f} GB/s" if metrics['hb_bw'] is not None else "—"

            head2 = ["PE (fpus)", "FPU Freq", "HB Bandwidth", "Input KB", "Weight KB", "Output KB", "NMP Ch"]
            rows2 = [[fpu, freq, hb, ib, wb, ob, nmp]]
            widths2 = [max(len(h), max(len(str(r[i])) for r in rows2)) for i, h in enumerate(head2)]

            head3 = ["Metric", "All Latency (s)"]
            rows3 = [
                ["Layer Number", f"{metrics['layer_num']}" if metrics['layer_num'] is not None else "—"],
                ["Token Number", f"{metrics['token_num']}" if metrics['token_num'] is not None else "—"],
                ["TTFT", f"{metrics['ttft']:.12f}" if metrics['ttft'] is not None else "—"],
                ["TPOT", f"{metrics['tpot']:.12f}" if metrics['tpot'] is not None else "—"],
                ["End-to-end Latency", f"{metrics['end_to_end_latency']:.12f}" if metrics['end_to_end_latency'] is not None else "—"],
                ["Goodput", f"{metrics['goodput']:.12f}" if metrics['goodput'] is not None else "—"],
                ["Throughput", f"{metrics['throughput']:.12f}" if metrics['throughput'] is not None else "—"],
            ]
            widths3 = [max(len(h), max(len(str(r[i])) for r in rows3)) for i, h in enumerate(head3)]


            banner = f" {title} "
            total_w = sum(widths1) + 3 + 3  
            print(_mk_line([total_w], ch='='))
            print('|' + banner.center(total_w) + '|')
            print(_mk_line([total_w], ch='='))

            print(_mk_line(widths1))
            print(_mk_row(widths1, head1))
            print(_mk_line(widths1))
            for r in rows1: print(_mk_row(widths1, r))
            print(_mk_line(widths1))
            print()  

            print(_mk_line(widths2))
            print(_mk_row(widths2, head2))
            print(_mk_line(widths2))
            for r in rows2: print(_mk_row(widths2, r))
            print(_mk_line(widths2))
            print()

            print(_mk_line(widths3))
            print(_mk_row(widths3, head3))
            print(_mk_line(widths3))
            for r in rows3: print(_mk_row(widths3, r))
            print(_mk_line(widths3))
        
        if self.need_eval_model_type == 'llama-lora': 
            kick_the_tires = 'llama3.1-8b-lora'

        elif self.need_eval_model_type == 'qwen-lora': 
            kick_the_tires = 'qwen2.5-1.5b-lora'
        
        elif self.need_eval_model_type == 'llama3.2-3b-lora' : 
            kick_the_tires = 'llama3.2-3b-lora'  
        elif self.need_eval_model_type == 'esm2-3b-lora' :
            kick_the_tires = 'llama3.2-3b-lora'

        log_dir = Path(ROOT_DSE) / "kick_the_tires" / kick_the_tires
        print(log_dir)
        # /home/myh/H2-LLM-ISCA-2025/hw_nas/kick_the_tires/result.log

        m = parse_result_log(log_dir)
        print_tables(m, title=f"/ kick_the_tires / {kick_the_tires}")

        #######################
        #      LoRA+FT     
        #######################

        # /home/users/tomomasa_yamasaki/H2-LLM-ISCA-2025-TForge-DeepSeek-R1-Distill-Llama-8B/lora_with_llm/try_this/esm/sweet_spot_run_lora_screener_GA.py
        from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft
        # from lora_with_llm.try_this.sweet_spot_run_lora_screener import make_lora_ft
        
        ppl, delt_ppl=make_lora_ft('searchspace/configs_single.json', self.model_id)

        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "ppl": ppl,
        #     "delt_ppl": delt_ppl,
        # })
        
        return ppl, m['end_to_end_latency']

    def _rbflex_and_dse_lat_only(self, image, candi_network, candi_hardparams):

        self.estm_vist_time += 1

        c_list = candi_network.split(':')

        # build searchspace--->.json
        HERE = Path(__file__).resolve()
        ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
        sys.path.insert(0, str(ROOT_LORA))

        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
        make_searchspace(c_list[0], c_list[1], c_list[2], c_list[3], self.seed)
                        #  pos,        r,        prec,        layer=0,        sd=42
        
        # print(candi_network.split(':'))
        # print(type(candi_network.split(':')))
        # for item in candi_network.split(':'):
        #     print(item)
        #     print(type(item))

        c_list = candi_network.split(':')

        #######################
        #         DSE         
        #######################
        ROOT_DSE= Path(__file__).resolve().parents[2]   # …/H2-LLM-ISCA-2025
        if str(ROOT_DSE) not in sys.path:
            sys.path.insert(0, str(ROOT_DSE))

        from main import make_parse_or_dse

        assert self.model_id is not None, "Check model_id!!"

        need_eval_model_id = self.model_id.split("/")[1]

        # Llama-3.2-3B / Qwen2.5-1.5B / Llama-3.1-8B
        if need_eval_model_id== 'Llama-3.1-8B': 
            self.need_eval_model_type = 'llama-lora'

        elif need_eval_model_id == 'Qwen2.5-1.5B ': 
            self.need_eval_model_type = 'qwen-lora'
        
        elif need_eval_model_id == 'Llama-3.2-3B': 
            self.need_eval_model_type = 'llama3.2-3b-lora'        
        elif need_eval_model_id == 'esm2_t36_3B_UR50D':
            self.need_eval_model_type = 'esm2-3b-lora'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. 转为 list[int]
        net_list = [int(float(x)) for x in candi_network.split(":")]   # [3, 7, 2, 9]

        # 3. 变成 2D Tensor，带 tkwargs（device, dtype）
        net_tensor  = torch.tensor([net_list], **tkwargs)          # shape: (1, 4)
        hard_tensor = torch.tensor([candi_hardparams], **tkwargs)  # shape: (1, 6)

        # 4. 在 Tensor 上 snap 到合法档位
        #    注意 cols 范围要覆盖到所有列：range(4) 和 range(6)
        net_tensor  = self.snap_to_levels_batch(net_tensor,  levels_map=self.LEVELS_sw, cols=range(4))
        hard_tensor = self.snap_to_levels_batch(hard_tensor, levels_map=self.LEVELS_hw, cols=range(6))

        # 5. 把 snapped Tensor 再转回 Python 类型
        #    这里建议先 detach().cpu() 再转 int 和 list
        net_list_snapped = (
            net_tensor
            .detach()
            .cpu()
            .squeeze(0)        # (1,4) -> (4,)
            .to(torch.int64)
            .tolist()
        )
        hardparams_snapped = (
            hard_tensor
            .detach()
            .cpu()
            .squeeze(0)        # (1,6) -> (6,)
            .to(torch.int64)
            .tolist()
        )

        # 6. 还原成你原来的两种变量形式
        candi_network    = ":".join(str(v) for v in net_list_snapped)   # "a:b:c:d"
        candi_hardparams = hardparams_snapped                           # list[int]

        print("\033[92m[ok]\033[0m")
        pprint({
            "candi_network": candi_network,
            "candi_hardparams": candi_hardparams,
        })

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # candi_network, candi_hardparams
        make_parse_or_dse('parse', candi_hardparams[0], candi_hardparams[1], candi_hardparams[2], candi_hardparams[3], candi_hardparams[4], candi_hardparams[5], self.need_eval_model_type)
        make_parse_or_dse('dse', candi_hardparams[0], candi_hardparams[1], candi_hardparams[2], candi_hardparams[3], candi_hardparams[4], candi_hardparams[5], self.need_eval_model_type)

        def parse_result_log(log_dir: Path):
            # /home/myh/H2-LLM-ISCA-2025/hw_nas/kick_the_tires/llama3.2-3b-lora

            log = sorted((log_dir).rglob("result.log"), key=lambda p: p.stat().st_mtime)[-1]
            text = log.read_text(encoding="utf-8")

            def grab(pattern, cast=float, default=None):
                m = re.search(pattern, text)
                if not m: return default
                return cast(m.group(1))

            idv     = grab(r"Best idv latency is ([0-9.eE+-]+)")
            prefill = grab(r"Prefill latency is ([0-9.eE+-]+)")
            decode  = grab(r"Decoding latency is ([0-9.eE+-]+)")

            layer_num = grab(r"Total layer num is ([0-9.eE+-]+)")
            token_num = grab(r"Total tokens is ([0-9.eE+-]+)")
            ttft = grab(r"TTFT is ([0-9.eE+-]+)")
            tpot = grab(r"TPOT is ([0-9.eE+-]+)")
            end_to_end_latency = grab(r"End-to-end Latency is ([0-9.eE+-]+)")
            goodput = grab(r"Goodput is ([0-9.eE+-]+)")
            throughput = grab(r"Throughput is ([0-9.eE+-]+)")

            m = re.search(r"PE config\s+(\d+)\s+fpus\s*@\s*([0-9.]+)GHz,\s*hybrid bonding bandwidth\s*([0-9.]+)GB/s", text)
            fpu, freq, hb_bw = (int(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)

            m = re.search(r"Input buffer size\s*([0-9.]+)KB,\s*weight buffer size\s*([0-9.]+)KB,\s*output buffer size\s*([0-9.]+)KB,\s*NMP channel num\s*([0-9]+)", text)
            ib_kb, wb_kb, ob_kb, nmp = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else (None, None, None, None)

            return dict(idv=idv, prefill=prefill, decode=decode,
                        fpu=fpu, freq=freq, hb_bw=hb_bw,
                        ib_kb=ib_kb, wb_kb=wb_kb, ob_kb=ob_kb, nmp=nmp, 
                        layer_num=layer_num, token_num=token_num, 
                        ttft=ttft, tpot=tpot, end_to_end_latency=end_to_end_latency,
                        goodput=goodput,throughput=throughput,)

        def _mk_line(widths, ch='-'):
            return '+' + '+'.join(ch*(w+2) for w in widths) + '+'

        def _mk_row(widths, cols):
            s = []
            for w, c in zip(widths, cols):
                s.append(' ' + str(c).ljust(w) + ' ')
            return '|' + '|'.join(s) + '|'

        def print_tables(metrics: dict, title="LLM Inference Summary"):
            head1 = ["Metric", "Pre Layer Latency (s)"]
            rows1 = [
                ["Best idv", f"{metrics['idv']:.12f}" if metrics['idv'] is not None else "—"],
                ["Prefill", f"{metrics['prefill']:.12f}" if metrics['prefill'] is not None else "—"],
                ["Decoding", f"{metrics['decode']:.12f}" if metrics['decode'] is not None else "—"],
            ]
            widths1 = [max(len(h), max(len(str(r[i])) for r in rows1)) for i, h in enumerate(head1)]

            ib = int(metrics['ib_kb']) if metrics['ib_kb'] is not None else "—"
            wb = int(metrics['wb_kb']) if metrics['wb_kb'] is not None else "—"
            ob = int(metrics['ob_kb']) if metrics['ob_kb'] is not None else "—"
            nmp = metrics['nmp'] if metrics['nmp'] is not None else "—"
            fpu = metrics['fpu'] if metrics['fpu'] is not None else "—"
            freq= f"{metrics['freq']:.1f} GHz" if metrics['freq'] is not None else "—"
            hb  = f"{metrics['hb_bw']:.1f} GB/s" if metrics['hb_bw'] is not None else "—"

            head2 = ["PE (fpus)", "FPU Freq", "HB Bandwidth", "Input KB", "Weight KB", "Output KB", "NMP Ch"]
            rows2 = [[fpu, freq, hb, ib, wb, ob, nmp]]
            widths2 = [max(len(h), max(len(str(r[i])) for r in rows2)) for i, h in enumerate(head2)]

            head3 = ["Metric", "All Latency (s)"]
            rows3 = [
                ["Layer Number", f"{metrics['layer_num']}" if metrics['layer_num'] is not None else "—"],
                ["Token Number", f"{metrics['token_num']}" if metrics['token_num'] is not None else "—"],
                ["TTFT", f"{metrics['ttft']:.12f}" if metrics['ttft'] is not None else "—"],
                ["TPOT", f"{metrics['tpot']:.12f}" if metrics['tpot'] is not None else "—"],
                ["End-to-end Latency", f"{metrics['end_to_end_latency']:.12f}" if metrics['end_to_end_latency'] is not None else "—"],
                ["Goodput", f"{metrics['goodput']:.12f}" if metrics['goodput'] is not None else "—"],
                ["Throughput", f"{metrics['throughput']:.12f}" if metrics['throughput'] is not None else "—"],
            ]
            widths3 = [max(len(h), max(len(str(r[i])) for r in rows3)) for i, h in enumerate(head3)]


            banner = f" {title} "
            total_w = sum(widths1) + 3 + 3  
            print(_mk_line([total_w], ch='='))
            print('|' + banner.center(total_w) + '|')
            print(_mk_line([total_w], ch='='))

            print(_mk_line(widths1))
            print(_mk_row(widths1, head1))
            print(_mk_line(widths1))
            for r in rows1: print(_mk_row(widths1, r))
            print(_mk_line(widths1))
            print()  

            print(_mk_line(widths2))
            print(_mk_row(widths2, head2))
            print(_mk_line(widths2))
            for r in rows2: print(_mk_row(widths2, r))
            print(_mk_line(widths2))
            print()

            print(_mk_line(widths3))
            print(_mk_row(widths3, head3))
            print(_mk_line(widths3))
            for r in rows3: print(_mk_row(widths3, r))
            print(_mk_line(widths3))
        
        if self.need_eval_model_type == 'llama-lora': 
            kick_the_tires = 'llama3.1-8b-lora'

        elif self.need_eval_model_type == 'qwen-lora': 
            kick_the_tires = 'qwen2.5-1.5b-lora'
        
        elif self.need_eval_model_type == 'llama3.2-3b-lora' : 
            kick_the_tires = 'llama3.2-3b-lora'  
        elif self.need_eval_model_type == 'esm2-3b-lora' :
            kick_the_tires = 'llama3.2-3b-lora'

        log_dir = Path(ROOT_DSE) / "kick_the_tires" / kick_the_tires
        print(log_dir)
        # /home/myh/H2-LLM-ISCA-2025/hw_nas/kick_the_tires/result.log

        m = parse_result_log(log_dir)
        print_tables(m, title=f"/ kick_the_tires / {kick_the_tires}")

        #######################
        #      LoRA+FT     
        #######################
        
        # build searchspace--->.json
        HERE = Path(__file__).resolve()
        ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
        sys.path.insert(0, str(ROOT_LORA))

        return m['end_to_end_latency']

    def _generate_initial_data(self, image, n):
        print("==> Generate initial data for optimization..")
        if self.acqu_algo == "Coflex" or self.acqu_algo == "Coflex_ppl":
            print("- Sampling method - LHS - ")
            
            import matplotlib.pyplot as plt
            # import seaborn as sns
            from scipy.stats import qmc
            # from sklearn.decomposition import PCA

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False 
            N = self.n_init_size  # Initial sample quantity
            dim = 10  # Total input dimension 4+6

            # Latin Hypercube sampling
            sampler = qmc.LatinHypercube(d=dim)
            samples = sampler.random(n=N)

            def map_to_levels(u, levels: np.ndarray) -> np.ndarray:
                L = len(levels)
                idx = np.floor(u * L).astype(int)
                idx = np.clip(idx, 0, L - 1)
                return levels[idx]

            # 1) sw: pos {0..11} (6 singles + 6 expert combos)
            sw_pos_levels = self.LEVELS_sw[0].astype(int)
            sw_pos = map_to_levels(samples[:, 0], sw_pos_levels)

            # 2) sw: rank {0,1,2,4,8,16,32,64,128,256,512}
            sw_rank_levels = np.array([0,1,2,4,8,16,32,64,128,256,512], dtype=int)
            sw_rank = map_to_levels(samples[:, 1], sw_rank_levels)

            # 3) sw: precision {2}
            # sw_prec_levels = np.arange(0, 2, 1, dtype=int)
            sw_prec_levels = np.array([2], dtype=int)
            sw_prec = map_to_levels(samples[:, 2], sw_prec_levels)

            # 4) sw: layer {0..44} (0~35 singles + 36~44 expert combos)
            sw_layer_levels = self.LEVELS_sw[3].astype(int)
            sw_layer = map_to_levels(samples[:, 3], sw_layer_levels)

# -------------------------------------------------------------------

            # 5) hw: precision {2}
            hw_prec_levels = np.array([2], dtype=int)
            hw_prec = map_to_levels(samples[:, 4], hw_prec_levels)
            
            # 6) hw: nmp_channel_num-space {2,4,6,8}  
            hw_chan_levels = np.array([2,4,6,8], dtype=int)
            hw_chan = map_to_levels(samples[:, 5], hw_chan_levels)

            # 7~9) hw: i/w/o_buffer_size {4,8,16,32,64,128,256,512,1024}  
            hw_buffer_size_levels = np.array([4,8,16,32,64,128,256,512,1024], dtype=int)
            hw_buffer_size = map_to_levels(samples[:, 6:9], hw_buffer_size_levels)   # (N,3)

            # 10) hw: fpu_pe_bw_space {0..15}
            hw_bw_space_levels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=int)
            hw_bw_space = map_to_levels(samples[:, 9], hw_bw_space_levels)

            design_samples = np.column_stack([
                sw_pos, sw_rank, sw_prec, sw_layer, 
                hw_prec,
                hw_chan,                    
                hw_buffer_size,              
                hw_bw_space                     
            ]).astype(int, copy=False)

            print("\033[92m[ok]\033[0m")
            print(design_samples.shape)
            for i in range(10):   
                print({i: np.unique(design_samples[:, i])}, end='\n')
        else:
            print("- Sampling method - Sobol - ")

        def process_candidate(candidate, arch_func):
            """Helper function to process a candidate."""
            
            arch = arch_func(candidate) # pos | rank | prec_sw | layer |
            accelerator = candidate[4:] 

            # print("\033[92m[ok]\033[0m")
            # pprint({
            #     "candidate": candidate,
            #     "arch": arch,
            #     "accelerator": accelerator,
            # })

            if self.Hardware_Arch == "DeFiNES":
                err, energy, cycle, EDP = self._rbflex_and_dse(image, arch, accelerator)
                return [err, EDP]
            
            elif self.Hardware_Arch == "ScaleSim":
                with torch.no_grad():
                    err, cycle = self._rbflex_and_dse(image, arch, accelerator)
                return [err, cycle]
                                    
            elif self.Hardware_Arch == "H^2":

                ppl, lat = self._rbflex_and_dse(image, arch, accelerator)

                print("\033[92m[ok]\033[0m")
                pprint({
                    "ppl": ppl,
                    "lat": lat,
                })
                self.hard_free()
                return [ppl, lat]
            
        if self.benchmark == "sss":
            if self.acqu_algo == "Coflex" or self.acqu_algo == "Coflex_ppl":
                train_x = torch.tensor(design_samples, **tkwargs)
            else:
                train_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds, n=n, q=1).squeeze(1))
                train_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1))
                train_x = torch.cat((train_x_sf, train_x_hd), 1)
            
            num_candidates = train_x.size(0)
            # Preallocate train_obj for performance
            train_obj = []
            for i in tqdm(range(num_candidates), ncols=80):
                candidate = train_x[i].int().tolist()
                arch_func = lambda c: '{}:{}:{}:{}:{}'.format(c[0], c[1], c[2], c[3], c[4])
                train_obj.append(process_candidate(candidate, arch_func))
            train_obj = torch.tensor(train_obj, **tkwargs)
        
        elif self.benchmark == "lora" or self.benchmark == 'qwen-lora':
            train_x = torch.tensor(design_samples, **tkwargs)
            num_candidates = train_x.size(0)

            # Preallocate train_obj for performance
            train_obj = []
            for i in tqdm(range(num_candidates), ncols=80):
                candidate = train_x[i].int().tolist()
                arch_func = lambda c: '{}:{}:{}:{}'.format(c[0], c[1], c[2], c[3]) # pos | rank | prec_sw | layer
                train_obj.append(process_candidate(candidate, arch_func))
            train_obj = torch.tensor(train_obj, **tkwargs)

        elif self.benchmark == "201":
            # Placeholder logic for benchmark 201
            # train_obj = torch.empty(0)
            # train_x = train_x_hd
            train_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1))
            train_x = torch.cat((train_x_sf, train_x_hd), 1)
            num_candidates = train_x.size(0)
            # Preallocate train_obj for performance
            train_obj = []
            for i in tqdm(range(num_candidates), ncols=80):
                candidate = train_x[i].int().tolist()
                arch_func = lambda c: '{}:{}:{}:{}:{}'.format(c[0], c[1], c[2], c[3], c[4])
                train_obj.append(process_candidate(candidate, arch_func))
            train_obj = torch.tensor(train_obj, **tkwargs)

        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")
        return train_x, train_obj
    
    def _initialize_model(self, train_x, train_obj, bounds):
        train_x = train_x.to(**tkwargs)
        train_y = train_obj.to(**tkwargs)

        if bounds is not None:
            bounds = bounds.to(train_x)

        models = []
        for i in range(train_y.shape[-1]):
            yi = train_y[..., i:i+1]

            if yi.std() < 1e-8:
                yi = yi - yi.mean()  
                yi = yi + 1e-6 * torch.randn_like(yi)  # very small jitter

            models.append(
                SingleTaskGP(
                    train_X= train_x,
                    train_Y= yi,
                    input_transform= Normalize(d=train_x.shape[-1], bounds=bounds),
                    outcome_transform= Standardize(m=1),
                )
            )
        model = ModelListGP(*models)
        mll   = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def pareto_dominance_check(self, y_err, y_eng):
        population = np.column_stack((y_err, y_eng))
        def dominates(p, q):
            return np.all(p <= q) and np.any(p < q)
        def fast_non_dominated_sort(population):
            S = [[] for _ in range(len(population))]
            n = np.zeros(len(population)) 
            fronts = [[]]
            rank = np.zeros(len(population)) 
            for p in range(len(population)):
                for q in range(len(population)):
                    if dominates(population[p], population[q]):
                        S[p].append(q)
                    elif dominates(population[q], population[p]):
                        n[p] += 1
                if n[p] == 0:
                    rank[p] = 0
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1], rank
    
    
        fronts, _ = fast_non_dominated_sort(population)
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        # if len(fronts) <= 6:
        #     for i, front in enumerate(fronts):
        #         front_points = population[front]
        #         plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=100, edgecolor='k', alpha=0.6)
        # else:
        #     group_size = 6
        #     groups = [fronts[i:i + group_size] for i in range(0, len(fronts), group_size)]
        #     selected_fronts = [group[0] for group in groups]
        #     for i, front in enumerate(selected_fronts):
        #         front_points = population[front]
        #         plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=100, edgecolor='k', alpha=0.6)
        for i, front in enumerate(fronts):
            front_points = population[front]
            plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=50, edgecolor='k', alpha=0.6)
        plt.title('Pareto Fronts Visualization', fontsize=9)
        plt.style.use('ggplot')
        plt.xlabel('Error (y_err)', fontsize=9)
        plt.ylabel('Energy (y_eng)', fontsize=9)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.savefig('pareto fronts visualization.png', dpi=300)
        # plt.show()
        return fronts, population
    """[Todo]Check this program."""
    
    def flatten_ranked_score_list(self, ranked_score_list):
        configs = []
        proxy_scores = []
        for item in ranked_score_list:
            # item: list[dict]
            assert isinstance(item, list) and len(item) >= 1
            d = item[0]  # 你现在单点 LoRA 就取第一个；多点你也可以自己 merge
            configs.append(d)
            proxy_scores.append(float(d["proxy_score"]))
        return configs, proxy_scores

    def _get_new_data(self, image, acqu_algo, model, train_x, train_obj, sampler):

        # if acqu_algo not in ("random", "random_ppl", "Coflex", "Coflex_ppl"):
        #     self.set_all_seeds(self.seed)

        AF_flag = True
        if acqu_algo == "qNEHVI" or acqu_algo == "qNEHVI_ppl":
            candidates = self._optimize_qnehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qEHVI" or acqu_algo == "qEHVI_ppl":
            candidates = self._optimize_qehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qNParEGO" or acqu_algo == "qNParEGO_ppl":
            candidates = self._optimize_qnparego_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "Coflex" or acqu_algo == "Coflex_ppl":

            print("\033[92m[ok]\033[0m")
            pprint({
                "self.status": self.status,
            })

            if self.status == "ERR":
                assert isinstance(train_x, np.ndarray), "train_x should be a NumPy array"
                assert isinstance(train_obj, np.ndarray), "train_obj should be a NumPy array"
                assert train_x.shape[1] == 4, "train_x should have 4 columns"
                assert train_obj.shape[1] == 1, "train_obj should have 1 column"
                assert train_x.shape[0] == train_obj.shape[0], "train_x and train_obj should have the same number of rows"
                assert train_x.ndim == 2, "train_x should be a 2D array"
                assert train_obj.ndim == 2, "train_obj should be a 2D array"

                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "self.bounds_err": self.bounds_err,
                #     "self.bounds_forstard_err": self.bounds_forstard_err,
                # })
                
                kernel = C(1.0, (1e-8, 1e8)) * RBF(length_scale=32, length_scale_bounds=(8, 64))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                scaler = StandardScaler().fit(train_x)
                train_x_scaled = scaler.transform(train_x)
                gp.fit(train_x_scaled, train_obj.ravel())
                
                bounds_err = np.array([
                    (0.0, 6.0),   # pos
                    (0.0, 512.0),  # rank
                    (0.0, 2.0),   # precision
                    (0.0, 15.0),   # layer
                ], dtype=float)

                candidates = []
                y_min_err = torch.min(torch.tensor(train_obj)).item()
                sw_levels = [self.LEVELS_sw[i] for i in range(4)]
                existing = {tuple(map(int, row)) for row in np.asarray(train_x, dtype=int)}
                for _ in range(self.BATCH_SIZE):
                    selected = None
                    for _try in range(8):
                        candidate = self.optimize_acquisition(
                            gp,
                            train_x,
                            bounds_err,
                            y_min_err,
                            scaler=scaler,
                            levels=sw_levels,
                        ).reshape(1, -1)
                        cand_t = torch.tensor(candidate, **tkwargs)
                        cand_t = self.snap_to_levels_batch(cand_t, levels_map=self.LEVELS_sw, cols=range(4))
                        cand = tuple(int(v) for v in cand_t.squeeze(0).cpu().numpy().tolist())
                        if cand not in existing:
                            selected = np.asarray(cand, dtype=int).reshape(1, -1)
                            break
                    if selected is None:
                        for _random_try in range(64):
                            random_row = np.array([np.random.choice(levels) for levels in sw_levels], dtype=int).reshape(1, -1)
                            cand = tuple(int(v) for v in random_row.squeeze(0).tolist())
                            if cand not in existing:
                                selected = random_row
                                break
                    if selected is None:
                        selected = np.array([np.random.choice(levels) for levels in sw_levels], dtype=int).reshape(1, -1)
                    existing.add(tuple(int(v) for v in selected.squeeze(0).tolist()))
                    candidates.append(selected)

                new_x_np = np.vstack(candidates)
                new_x = torch.tensor(new_x_np, **tkwargs)

            elif self.status == "ENG":
                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "self.bounds_eng": self.bounds_eng,
                #     "self.bounds_forstard_eng": self.bounds_forstard_eng,
                # })
                new_x = self._optimize_coflex_and_get_observation(model, train_x, sampler, self.bounds_eng, self.bounds_forstard_eng)
                eps = 1e-6
                lo, hi = self.bounds_eng[0], self.bounds_eng[1]
                new_x = new_x.clamp(lo + eps, hi - eps)
                new_x = self.snap_to_levels_batch(new_x, levels_map=self.LEVELS_hw, cols=range(6))

            elif self.status == "NOM":
                new_x = self._optimize_qnparego_and_get_observation(model, train_x, sampler)
                eps = 1e-6
                lo, hi = self.bounds[0], self.bounds[1]
                new_x = new_x.clamp(lo + eps, hi - eps)

                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "candidates": candidates,
                # })

        elif acqu_algo == "nsga" or acqu_algo == "nsga_ppl":
            candidates = self._optimize_nsga_and_get_observation(train_x, train_obj, self.bounds_fornorm, self.image, self.BATCH_SIZE, self)
            print(f"[NSGA] archive size = {len(self.nsga_archive)}, hits={self.cache_hits}, miss={self.cache_miss}")

        elif acqu_algo == "random" or acqu_algo == "random_ppl":
            AF_flag = False
            new_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
            new_x_hd = self.snap_to_levels_batch(new_x_hd, levels_map=self.LEVELS_hw, cols=range(6))

            if self.benchmark == "lora" or self.benchmark == 'qwen-lora':
                new_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x_sf = self.snap_to_levels_batch(new_x_sf, levels_map=self.LEVELS_sw, cols=range(4))

                new_x = torch.cat((new_x_sf, new_x_hd), 1)

                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "self.sf_bounds": self.sf_bounds,
                #     "self.hd_bounds": self.hd_bounds,
                #     "new_x_sf": new_x_sf,
                #     "new_x_hd": new_x_hd,
                #     "new_x": new_x,
                # })

            elif self.benchmark == "sss":
                new_x_sf = 8*torch.floor(draw_sobol_samples(self.sf_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)
            
            elif self.benchmark == "201":
                new_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)

        elif acqu_algo == "SWoptOnlyrandom_ppl":
            AF_flag = False
            if not (self.benchmark == "lora" or self.benchmark == "qwen-lora"):
                raise ValueError("SWoptOnlyrandom_ppl currently supports lora/qwen-lora benchmark only.")

            new_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds, n=self.BATCH_SIZE, q=1).squeeze(1))
            new_x_sf = self.snap_to_levels_batch(new_x_sf, levels_map=self.LEVELS_sw, cols=range(4))

            global GLOBAL_SW_OPT_ONLY_FIXED_HW
            if GLOBAL_SW_OPT_ONLY_FIXED_HW is None:
                GLOBAL_SW_OPT_ONLY_FIXED_HW = np.array(
                    [np.random.choice(self.LEVELS_hw[i]) for i in range(6)],
                    dtype=int,
                )

            fixed_hw = torch.tensor(GLOBAL_SW_OPT_ONLY_FIXED_HW, **tkwargs).unsqueeze(0).repeat(self.BATCH_SIZE, 1)
            new_x = torch.cat((new_x_sf, fixed_hw), 1)

        elif acqu_algo == "HWoptOnlyrandom_ppl":
            AF_flag = False
            if not (self.benchmark == "lora" or self.benchmark == "qwen-lora"):
                raise ValueError("HWoptOnlyrandom_ppl currently supports lora/qwen-lora benchmark only.")

            new_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds, n=self.BATCH_SIZE, q=1).squeeze(1))
            new_x_hd = self.snap_to_levels_batch(new_x_hd, levels_map=self.LEVELS_hw, cols=range(6))

            fixed_sw = torch.zeros((self.BATCH_SIZE, 4), **tkwargs)
            new_x = torch.cat((fixed_sw, new_x_hd), 1)
        else:
            print("Select correct acquation function from [qNEHVI, qEHVI, qNParEGO, random, nsga, SWoptOnlyrandom_ppl, HWoptOnlyrandom_ppl]")
            exit()

        if AF_flag:
            if acqu_algo == "Coflex" or acqu_algo == "Coflex_ppl":
                if self.status == "NOM":
                    # new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds))
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "nom_new_x": new_x,
                    })     
                elif self.status == "ERR":
                    pass
                    # new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_err))
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "sw_new_x": new_x,
                    })
                elif self.status == "ENG":
                    # new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_eng))
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "hw_new_x": new_x,
                    })
            else: # Other MOBO(AF)
                candidates = candidates.to(self.bounds.device)
                new_x = torch.floor(unnormalize(candidates.detach(), bounds=self.bounds))
                new_x = self.snap_to_levels_batch(new_x, levels_map=self.LEVELS, cols=range(10))
                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "acqu_algo": acqu_algo,
                #     "candidates": candidates,
                #     "nom_new_x": new_x,
                #     "new_x": new_x,
                # })
        
        if self.benchmark == "lora" or self.benchmark == "qwen-lora":
            if self.status == "NOM":
                assert new_x.shape[1] == 10, "new_x should have 6 columns"
                assert new_x.shape[0] == self.BATCH_SIZE, "new_x should have the same number of rows as self.BATCH_SIZE"
                assert new_x.ndim == 2, "new_x should be a 2D tensor"
            elif self.status == "ERR":
                assert new_x.shape[1] == 4, "new_x should have 4 columns"
                assert new_x.shape[0] == self.BATCH_SIZE, "new_x should have the same number of rows as self.BATCH_SIZE"
                assert new_x.ndim == 2, "new_x should be a 2D tensor"
            elif self.status == "ENG":
                assert new_x.shape[1] == 6, "new_x should have 6 columns"
                assert new_x.shape[0] == self.BATCH_SIZE, "new_x should have the same number of rows as self.BATCH_SIZE"
                assert new_x.ndim == 2, "new_x should be a 2D tensor"
        
        if self.status == "NOM":
            new_obj = []
            new_x = self.snap_to_levels_batch(new_x, levels_map=self.LEVELS, cols=range(10))
            lat_list = []
            # assert new_x.shape[1] == 10, "x should have 10 cols"
            if self.is_score_based:
                for c in new_x.tolist():
                    if self.Hardware_Arch == "H^2":
                        arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                        accelerator = [int(round(x)) for x in c[4:]]
                        print("\033[92m[ok]\033[0m")
                        pprint({
                            "nom_cand": c,
                        })
                        
                        #将求score单独拎出来, 先整部分score
                        HERE = Path(__file__).resolve()
                        ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
                        sys.path.insert(0, str(ROOT_LORA))

                        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                        make_searchspace(c[0], c[1], c[2], c[3],self.seed)
                                        #  pos,  r,  prec, layer=0,        sd=42

                        # /home/myh/H2-LLM-ISCA-2025/lora_with_llm/try_this/proxy.py
                        from lora_with_llm.try_this.proxy import predict_from_json_file

                        ranked = predict_from_json_file("proxy_ckpt.pt", "searchspace/configs_single.json", device=DEVICE)
                        # print(ranked)

                        self.arch_list.append(arch)
                        self.ranked_score_list.append(ranked)

                        self.hard_free()

                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        #这里_rbflex_and_dse改成仅提供 lat
                        lat = self._rbflex_and_dse_lat_only(image, arch, accelerator)
                        lat_list.append(lat)

                        # self.cost_time = time.time() - self.start_time
                        # if self.search_time_rec is None:
                        #     self.search_time_rec = torch.tensor([[ppl, lat, self.cost_time, self.iter]], **tkwargs)
                        # else:
                        #     self.search_time_rec = torch.cat(
                        #         (self.search_time_rec, 
                        #         torch.tensor([[ppl, lat, self.cost_time, self.iter]], **tkwargs)
                        #         )
                        #         , dim=0)
                        
                    elif self.Hardware_Arch == "DeFiNES":
                        if self.benchmark == "sss" or self.benchmark == "201":
                            candidate = list(map(int, candidate))
                            arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3],
                                                        candidate[4])
                            # print("this network arch: " + str(candidate[0]) + "," + str(candidate[1]) + "," + str(
                            #     candidate[2]) + "," + str(candidate[3]) + "," + str(candidate[4]))
                            accelerator = candidate[5:]
                        err, energy, cycle, EDP = self._rbflex_and_dse(image, arch, accelerator)
                        self.cost_time = time.time() - self.start_time
                        if self.search_time_rec is None:
                            self.search_time_rec = torch.tensor([[err, EDP, self.cost_time, self.iter]], **tkwargs)
                        else:
                            self.search_time_rec = torch.cat(
                                (self.search_time_rec, 
                                torch.tensor([[err, EDP, self.cost_time, self.iter]], **tkwargs)
                                )
                                , dim=0)             
                    elif self.Hardware_Arch == "ScaleSim":
                        if self.benchmark == "sss":
                            candidate = list(map(int, candidate))
                            arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3],
                                                        candidate[4])
                            # print("this network arch: " + str(candidate[0]) + "," + str(candidate[1]) + "," + str(
                            #     candidate[2]) + "," + str(candidate[3]) + "," + str(candidate[4]))
                            accelerator = candidate[5:]
                        err, cycle = self._rbflex_and_dse(image, arch, accelerator)

                # 这里开始计算整个批次score
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                self.arch_list = self.arch_list[-(self.BATCH_SIZE):]
                self.ranked_score_list = self.ranked_score_list[-(self.BATCH_SIZE):]

                print("\033[1;31m[ --------------------------------------------------- ranked score list --------------------------------------------------- ]\033[0m")
                pprint({
                    # "err_list": self.err_list,
                    "ranked_score_list": self.ranked_score_list,
                })

                from lora_with_llm.try_this.gate import gate_oracle
                oracle_indices, gate_stats, oracle_configs, configs, proxy_scores = gate_oracle(self.ranked_score_list, self.warm_up_train_x, self.global_oracle_ratio)

                self.total_candidates += len(configs)
                self.total_oracle += int(oracle_indices.numel())
                self.global_oracle_ratio = self.total_oracle / max(1, self.total_candidates)

                print("\033[1;31m[ --------------------------------------------------- Gate Status --------------------------------------------------- ]\033[0m")
                pprint({
                    "oracle_indices": oracle_indices, "gate_stats": gate_stats, "oracle_configs": oracle_configs,
                    "total_candidates": self.total_candidates,"total_oracle": self.total_oracle,"global_oracle_ratio": self.global_oracle_ratio,
                })
                
                y_true_list = []
                x_true_list = []
                x_true_tensor = None
                y_true_tensor = None
                
                # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.proxy_train_x, self.proxy_train_obj, device=DEVICE, save_path="Visualization_1")

                print(f"rho: {rho}")
                
                if len(oracle_configs) > 0:
                    for i in oracle_indices.tolist():

                        c = new_x[i] # c is tensor has 10 cols
                        
                        print("\033[92m[ok]\033[0m")
                        pprint({
                            "re-calibration_cand": c,
                        })
                        
                        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                        make_searchspace(c[0], c[1], c[2], c[3],self.seed)

                        from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft
                        
                        ppl, delt_ppl=make_lora_ft('searchspace/configs_single.json', self.model_id)
                        
                        # self.err_list.append(ppl)
                        x_true_list.append(c)
                        y_true_list.append(ppl)

                        self.hard_free()
                
                    x_true_tensor = torch.stack(x_true_list, dim=0)[:, :4]
                    # (K, 4)
                    y_true_tensor = torch.tensor(y_true_list, **tkwargs).view(-1, 1)   
                    # (K, 1)

                    assert x_true_tensor.shape[0] == y_true_tensor.shape[0], "x and y should have same rows"
                    assert x_true_tensor.shape[1] == 4, "x should have 4 cols"
                    assert y_true_tensor.shape[1] == 1, "y should have 1 cols"

                    # 累积所有“有真值”的点，作为 RankNet 的训练集
                    self.proxy_train_x = torch.cat([self.proxy_train_x, x_true_tensor], dim=0)
                    self.proxy_train_obj = torch.cat([self.proxy_train_obj, y_true_tensor], dim=0)

                    from lora_with_llm.try_this.proxy import fit_proxy_full, save_proxy_ckpt
                    # 用累积到目前为止的所有真值点重新训练 RankNet
                    self.proxy_model, self.proxy_std = fit_proxy_full(
                        self.proxy_train_x,      # (N_all, 4)
                        self.proxy_train_obj,    # (N_all, 1) 只包含 PPL
                        device=DEVICE,
                    )

                    # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                    from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                    rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.proxy_train_x, self.proxy_train_obj, device=DEVICE, save_path="Visualization_1")

                    # 用同一个 ckpt 路径覆盖，这样后面 Gate/Prio 直接加载到的是“最新版本”的 Proxy
                    ckpt_path = "proxy_ckpt.pt"   # 或者你在类里存的 self.proxy_ckpt_path
                    save_proxy_ckpt(ckpt_path, self.proxy_model, self.proxy_std, self.pos2id, self.prec2id)

                else:
                    # Skip if no re-calibration is triggered
                    pass

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                assert len(proxy_scores) == len(lat_list), "score和lat的数量需要相当"
                            # list

                for idx in range(len(proxy_scores)):
                    if self.Hardware_Arch == "H^2":
                        new_obj.append([proxy_scores[idx], lat_list[idx]])

                new_obj = torch.tensor(new_obj, **tkwargs)
                print("## The candidates ", new_x, new_obj)

                assert new_x.shape[0] == new_obj.shape[0], "x and y should have same rows"
                assert new_x.shape[1] == 10, "x should have 10 cols"
                assert new_obj.shape[1] == 2, "y should have 2 cols"
                
                return new_x, new_obj
            
            else:
                for c in new_x.tolist():
                    arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                    accelerator = [int(round(x)) for x in c[4:]]
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "nom_cand": c,
                    })

                    ppl, lat = self._rbflex_and_dse(image, arch, accelerator)
                    
                    self.hard_free()

                    if self.Hardware_Arch == "DL2":
                        new_obj.append([err, energy, cycle])
                    elif self.Hardware_Arch == "ScaleSim":
                        new_obj.append([err, cycle])       
                    elif self.Hardware_Arch == "DeFiNES":
                        new_obj.append([err, EDP])
                    elif self.Hardware_Arch == "H^2":
                        new_obj.append([ppl, lat])
        
                new_obj = torch.tensor(new_obj, **tkwargs)
                # print("## The candidates ", new_x, new_obj)
                return new_x, new_obj
     
        elif self.status == "ERR":
            new_x = self.snap_to_levels_batch(new_x, levels_map=self.LEVELS_sw, cols=range(4))
            
            if self.is_score_based:
                ckpt_path = "proxy_ckpt.pt"
                
                HERE = Path(__file__).resolve()
                ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
                sys.path.insert(0, str(ROOT_LORA))

                for c in new_x.tolist():
                    arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                    
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "err_cand": c,
                    })
                    
                    from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                    make_searchspace(c[0], c[1], c[2], c[3],self.seed)
                    
                    # /home/myh/H2-LLM-ISCA-2025/lora_with_llm/try_this/proxy.py
                    from lora_with_llm.try_this.proxy import predict_from_json_file
                    
                    # ranked: <class 'list'>
                    # ckpt_path = "proxy_ckpt.pt"
                    ranked = predict_from_json_file(ckpt_path, "searchspace/configs_single.json", device=DEVICE)

                    # from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft
                    # ppl, delt_ppl=make_lora_ft('searchspace/configs_single.json')
                    
                    # self.err_list.append(ppl)
                    self.arch_list.append(arch)
                    self.ranked_score_list.append(ranked)

                    self.hard_free()
        
                # self.err_list = self.err_list[-(self.BATCH_SIZE):]
                self.arch_list = self.arch_list[-(self.BATCH_SIZE):]
                self.ranked_score_list = self.ranked_score_list[-(self.BATCH_SIZE):]

                print("\033[1;31m[ --------------------------------------------------- ranked score list --------------------------------------------------- ]\033[0m")
                pprint({
                    # "err_list": self.err_list,
                    "ranked_score_list": self.ranked_score_list,
                })

                from lora_with_llm.try_this.gate import gate_oracle
                oracle_indices, gate_stats, oracle_configs, configs, proxy_scores = gate_oracle(self.ranked_score_list, self.warm_up_train_x, self.global_oracle_ratio)

                self.total_candidates += len(configs)
                self.total_oracle += int(oracle_indices.numel())
                self.global_oracle_ratio = self.total_oracle / max(1, self.total_candidates)

                print("\033[1;31m[ --------------------------------------------------- Gate Status --------------------------------------------------- ]\033[0m")
                pprint({
                    "oracle_indices": oracle_indices, "gate_stats": gate_stats, "oracle_configs": oracle_configs,
                    "total_candidates": self.total_candidates,"total_oracle": self.total_oracle,"global_oracle_ratio": self.global_oracle_ratio,
                })
                
                y_true_list = []
                x_true_list = []
                x_true_tensor = None
                y_true_tensor = None
                
                # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.proxy_train_x, self.proxy_train_obj, device=DEVICE, save_path="Visualization_2")
                
                if len(oracle_configs) > 0:
                    for i in oracle_indices.tolist():

                        c = new_x[i] # c is tensor with 4 cols
                        
                        print("\033[92m[ok]\033[0m")
                        pprint({
                            "re-calibration_cand": c,
                        })
                        
                        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                        make_searchspace(c[0], c[1], c[2], c[3],self.seed)

                        from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft

                        ppl, delt_ppl=make_lora_ft('searchspace/configs_single.json', self.model_id)

                        # self.err_list.append(ppl)
                        x_true_list.append(c)
                        y_true_list.append(ppl)

                        self.hard_free()
                
                    x_true_tensor = torch.stack(x_true_list, dim=0)
                    # (K, 4)
                    y_true_tensor = torch.tensor(y_true_list, **tkwargs).view(-1, 1)   
                    # (K, 1)

                    assert x_true_tensor.shape[0] == y_true_tensor.shape[0], "x and y should have same rows"
                    assert x_true_tensor.shape[1] == 4, "x should have 4 cols"
                    assert y_true_tensor.shape[1] == 1, "y should have 1 cols"

                    # 累积所有“有真值”的点，作为 RankNet 的训练集
                    self.proxy_train_x = torch.cat([self.proxy_train_x, x_true_tensor], dim=0)
                    self.proxy_train_obj = torch.cat([self.proxy_train_obj, y_true_tensor], dim=0)

                    from lora_with_llm.try_this.proxy import fit_proxy_full, save_proxy_ckpt
                    # 用累积到目前为止的所有真值点重新训练 RankNet
                    self.proxy_model, self.proxy_std = fit_proxy_full(
                        self.proxy_train_x,      # (N_all, 4)
                        self.proxy_train_obj,    # (N_all, 1) 只包含 PPL
                        device=DEVICE,
                    )

                    # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                    from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                    rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.proxy_train_x, self.proxy_train_obj, device=DEVICE, save_path="Visualization_2")

                    print(f"rho: {rho}")

                    # 用同一个 ckpt 路径覆盖，这样后面 Gate/Prio 直接加载到的是“最新版本”的 Proxy
                    ckpt_path = "proxy_ckpt.pt"   # 或者你在类里存的 self.proxy_ckpt_path
                    save_proxy_ckpt(ckpt_path, self.proxy_model, self.proxy_std, self.pos2id, self.prec2id)

                else:
                    # Skip if no re-calibration is triggered
                    pass

                self.err_list = proxy_scores

                return new_x, torch.tensor(proxy_scores).to(**tkwargs)
            else:
                for c in new_x.tolist():
                    arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "err_cand": c,
                    })
                    HERE = Path(__file__).resolve()
                    ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
                    sys.path.insert(0, str(ROOT_LORA))

                    from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                    make_searchspace(c[0], c[1], c[2], c[3],self.seed)
                    from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft

                    ppl, _=make_lora_ft('searchspace/configs_single.json', self.model_id)

                    self.err_list.append(ppl)
                    self.arch_list.append(arch)
                self.err_list = self.err_list[-(self.BATCH_SIZE):]
                self.arch_list = self.arch_list[-(self.BATCH_SIZE):]
                return new_x, torch.tensor(self.err_list).to(**tkwargs)

        elif self.status == "ENG":
            MIN_IDX = self.err_list.index(min(self.err_list))
            
            ADDIT_ERR = torch.tensor(([[[int(x) for x in self.arch_list[MIN_IDX].split(":")]]*self.BATCH_SIZE][0]), **tkwargs)
            ADDIT_ENG = new_x
            
            new_obj = []
            print("\033[92m[ok]\033[0m")
            pprint({
                "MIN_IDX": MIN_IDX,
                "ADDIT_ERR": ADDIT_ERR,
                "ADDIT_ENG": ADDIT_ENG,
            })
            ADDIT_ENG = self.snap_to_levels_batch(ADDIT_ENG, levels_map=self.LEVELS_hw, cols=range(6))

            new_x = torch.cat((ADDIT_ERR, ADDIT_ENG), dim=1)
            for c in new_x.tolist():                
                if self.Hardware_Arch == "H^2":
                    # arch = f"{c[0]}:{c[1]}:{c[2]}:{c[3]}" # pos:rank:prec:layer
                    # accelerator = c[4:]
                    arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                    accelerator = [int(round(x)) for x in c[4:]]
                    print("\033[92m[ok]\033[0m")
                    pprint({
                        "eng_cand": c,
                    })
                    ROOT_DSE= Path(__file__).resolve().parents[2]   # …/H2-LLM-ISCA-2025
                    if str(ROOT_DSE) not in sys.path:
                        sys.path.insert(0, str(ROOT_DSE))
                    
                    lat = self._rbflex_and_dse_lat_only(image, arch, accelerator)

                    new_obj.append(lat)
                        
            x = torch.as_tensor(new_obj, **tkwargs)
            x = x if x.ndim >= 2 else x.unsqueeze(0)
            new_obj = x.mT.squeeze(0)      
            # new_obj = torch.tensor(new_obj, **tkwargs).T.squeeze(0)

            return new_x[:, 4:], new_obj.to(**tkwargs)
    """[Todo] Solve the error. Float for the destination and Double for the source """
    

    def _optimize_qnehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        with torch.no_grad():
            Y = model.posterior(normalize(train_x.to(**tkwargs), self.bounds_fornorm.to(**tkwargs))).mean

        minimization_objective = GenericMCObjective(lambda Y, X=None: -Y)

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-self.ref_point,

            # objective=minimization_objective,
            
            X_baseline=normalize(train_x.to(**tkwargs), self.bounds_fornorm.to(**tkwargs)),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        
        return candidates
    
    
    def _optimize_qehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, self.bounds_fornorm)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=-self.ref_point, 
            Y=-pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=-self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return candidates
    
    
    def _optimize_qnparego_and_get_observation(self, model, train_x, sampler):
        
        # train_x = normalize(train_x.to(**tkwargs), self.bounds.to(**tkwargs))
        train_x = train_x.to(**tkwargs)
        
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        
        def ic_gen_with_ex(acq_function, bounds, q, num_restarts, raw_samples, options=None, **kwargs):
            # 1) 先生成默认 IC
            ics = gen_batch_initial_conditions(
                acq_function=acq_function,
                bounds=bounds,                 # 这里通常是 [0,1]^d 的 box（bounds_forstard）
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {"init_batch_limit": 100},
            )

            # 2) 若存在 EX，则注入为额外起点
            ex = getattr(self, "EX", None)
            if ex is None or ex.numel() == 0:
                return ics

            ex = ex.to(self.bounds.device, self.bounds.dtype)
            
            ex = self.snap_to_levels_batch(ex, levels_map=self.LEVELS, cols=range(10))
            
            eps = 1e-6
            lo = bounds[0].to(ex.device, ex.dtype)
            hi = bounds[1].to(ex.device, ex.dtype)
            ex = ex.clamp(lo + eps, hi - eps)

            # 按 num_restarts 预算截断，并扩展到 (r_ex, q, d)
            r_ex = min(ex.size(0), num_restarts)
            if r_ex > 0:
                ex_ic = ex[:r_ex].unsqueeze(1).expand(-1, q, -1)   # (r_ex, q, d)
                ex_ic = ex_ic.to(device=ics.device, dtype=ics.dtype)
                ics = torch.cat([ex_ic, ics], dim=0)                 # (r_ex + num_restarts, q, d)
            return ics

        ic_gen = ic_gen_with_ex

        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(self.mobo_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)

        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds,                 
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            ic_generator=ic_gen,                         
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates
    
    
    def _optimize_coflex_and_get_observation(self, model, train_x, sampler, bounds, bounds_forstard):
        
        # train_x = normalize(train_x, bounds)
        train_x = torch.tensor(train_x, **tkwargs)

        with torch.no_grad():
            pred = model.posterior(train_x).mean

        def _to_unit(levels, lo, hi):
            return (levels - lo) / (hi - lo + 1e-12)

        def build_levels_unit(bounds_2xd: torch.Tensor):
            lu = {}
            lo, hi = bounds_2xd[0].cpu().numpy(), bounds_2xd[1].cpu().numpy()
            if self.status == 'ERR':
                for j, levels in self.LEVELS_sw.items():
                    lu[j] = _to_unit(levels.astype(float), lo[j], hi[j])
            elif self.status == 'ENG':
                for j, levels in self.LEVELS_hw.items():
                    lu[j] = _to_unit(levels.astype(float), lo[j], hi[j])
            else:
                raise ValueError(f"Unsupported status: {self.status}")
            return lu
        
        def make_round_to_levels(levels_unit_dict):
            def round_to_levels(X: torch.Tensor) -> torch.Tensor:
                Xr = X.clone()
                for j, lv in levels_unit_dict.items():
                    lv_t = torch.as_tensor(lv, dtype=X.dtype, device=X.device)  # (L,)
                    dist = (Xr[..., j].unsqueeze(-1) - lv_t.unsqueeze(0)).abs()
                    idx  = dist.argmin(dim=-1)
                    Xr[..., j] = lv_t[idx]
                return Xr
            return round_to_levels

        def ic_gen_with_ex(acq_function, bounds, q, num_restarts, raw_samples, options=None, **kwargs):
            # 1) 先生成默认 IC
            ics = gen_batch_initial_conditions(
                acq_function=acq_function,
                bounds=bounds,                 # 这里通常是 [0,1]^d 的 box（bounds_forstard）
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {"init_batch_limit": 100},
            )

            # 2) 若存在 EX，则注入为额外起点
            ex = getattr(self, "EX", None)
            if ex is None or ex.numel() == 0:
                return ics
            if ex is not None:
                ex = ex[:, 4:]

            ex = ex.to(self.bounds.device, self.bounds.dtype)
            
            ex = self.snap_to_levels_batch(ex, levels_map=self.LEVELS_hw, cols=range(6))
            
            eps = 1e-6
            lo = bounds[0].to(ex.device, ex.dtype)
            hi = bounds[1].to(ex.device, ex.dtype)
            ex = ex.clamp(lo + eps, hi - eps)

            # 按 num_restarts 预算截断，并扩展到 (r_ex, q, d)
            r_ex = min(ex.size(0), num_restarts)
            if r_ex > 0:
                ex_ic = ex[:r_ex].unsqueeze(1).expand(-1, q, -1)   # (r_ex, q, d)
                ex_ic = ex_ic.to(device=ics.device, dtype=ics.dtype)
                ics = torch.cat([ex_ic, ics], dim=0)                 # (r_ex + num_restarts, q, d)
            return ics
        
        ic_gen = ic_gen_with_ex
        
        # LEVELS_UNIT = build_levels_unit(bounds_forstard)
        # round_to_levels = make_round_to_levels(LEVELS_UNIT)
        # ic_gen = make_ic_generator(round_to_levels)
        
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            # weights = sample_simplex(1, **tkwargs)[0] 
            # objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            # NOTE:
            # qNEI maximizes the acquisition objective by default.
            # Both ERR/PPL and ENG(latency) are minimization targets in this repo,
            # so we explicitly negate the scalarized objective here.
            # Without this sign flip, ENG tends to drift toward larger latency.
            weights = sample_simplex(1, **tkwargs)[0]
            scalarized = get_chebyshev_scalarization(weights=weights, Y=pred)
            objective = GenericMCObjective(lambda Y, X=None, s=scalarized: -s(Y, X))
            acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)

        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=bounds,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            ic_generator=ic_gen,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates
    
    
    def _optimize_nsga_and_get_observation(self, train_x, train_obj, bounds_fornorm, image, batch_size, outer):
        
        train_x = train_x.int()
        train_x = train_x.detach().cpu().numpy()   # (N, 10)
        train_obj = train_obj.detach().cpu().numpy()  # (N, 2)
        
        from pymoo.config import Config

        Config.warnings['not_compiled'] = False
        
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.visualization.scatter import Scatter
        import numpy as np
        from pymoo.core.callback import Callback
        from pymoo.indicators.hv import HV
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem
        from pymoo.core.individual import Individual
        from pymoo.core.evaluator import Evaluator

        from pymoo.problems.static import StaticProblem
        
        ##################################
        
        '''
        train_x: torch.Size([100, 10]) <class 'torch.Tensor'>
        train_obj: torch.Size([100, 2]) <class 'torch.Tensor'>
        bounds_fornorm: torch.Size([2, 10]) <class 'torch.Tensor'>
        '''

        class CustomProblem(Problem):
            def __init__(self, train_x, train_obj, bounds_fornorm, image, rbflex_method, is_score_based, outer):
                self.bounds_fornorm = bounds_fornorm.detach().cpu().numpy()

                self.train_x = train_x.detach().cpu().numpy()
                self.train_obj = train_obj.detach().cpu().numpy()
                
                self.rbflex_method = rbflex_method

                self.image = image

                self.is_score_based = is_score_based

                self.outer = outer   # outer 是 FRCN_Simulator 实例，里面有 nsga_archive 字段

                lb=bounds_fornorm[0].detach().cpu().numpy()
                ub=bounds_fornorm[1].detach().cpu().numpy()
                
                assert lb.shape == ub.shape, "lower and upper bound should have same cols"

                n_cols = lb.shape[0]

                for x_row, f_row in zip(self.train_x, self.train_obj):
                    key = tuple(int(v) for v in x_row)
                    if key not in self.outer.nsga_archive:
                        self.outer.nsga_archive[key] = (float(f_row[0]), float(f_row[1]))

                super().__init__(
                                n_var=n_cols,  # column of train_x as the numbers of vars
                                #  pos | rank | prec_sw | layer | prec_hw | nmp_c | i_buff | o_buff | w_buf |  bw |
                                n_obj=2,  # two objective targets
                                # ppl | lat
                                
                                n_constr=0,  # no constrain,
                                
                                xl=lb,
                                xu=ub,
                                
                                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                
                def get_network_score_edp(x):

                    x = list(map(int, x))

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 2. 转为 list[int]
                    all_list = x

                    all_tensor  = torch.tensor([all_list], **tkwargs)         

                    all_tensor  = self.outer.snap_to_levels_batch(all_tensor,  levels_map=self.outer.LEVELS, cols=range(10))

                    all_list_snapped = (
                        all_tensor
                        .detach()
                        .cpu()
                        .squeeze(0)       
                        .to(torch.int64)
                        .tolist()
                    )
 
                    x = all_list_snapped

                    # print("\033[92m[ok]\033[0m")
                    # pprint({
                    #     "nom_cand": x,
                    #     "type": type(x),
                    #     "len_nom_cand": len(x), 
                    # })
                    
                    lat_list = []

                    if self.is_score_based:

                        # arch = '{}:{}:{}:{}'.format(x[0], x[1], x[2], x[3]) # 4
                        # accelerator = x[4:] # 6
                        arch = f"{int(c[0])}:{int(c[1])}:{int(c[2])}:{int(c[3])}" # pos:rank:prec:layer
                        accelerator = [int(round(x)) for x in c[4:]]
                        
                        #将求score单独拎出来, 先整部分score
                        HERE = Path(__file__).resolve()
                        ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
                        sys.path.insert(0, str(ROOT_LORA))

                        from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                        make_searchspace(x[0], x[1], x[2], x[3],self.seed)
                                        #  pos,  r,  prec, layer=0,        sd=42

                        # /home/myh/H2-LLM-ISCA-2025/lora_with_llm/try_this/proxy.py
                        from lora_with_llm.try_this.proxy import predict_from_json_file

                        ranked = predict_from_json_file("proxy_ckpt.pt", "searchspace/configs_single.json", device=DEVICE)
                        # print(ranked)

                        self.outer.arch_list.append(arch)
                        self.outer.ranked_score_list.append(ranked)

                        self.outer.hard_free()

                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        #这里_rbflex_and_dse改成仅提供 lat
                        lat = self.outer._rbflex_and_dse_lat_only(image, arch, accelerator)
                        lat_list.append(lat)

                        # 这里开始计算整个批次score
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        self.outer.arch_list = self.outer.arch_list[-1:]
                        self.outer.ranked_score_list = self.outer.ranked_score_list[-1:]

                        print("\033[1;31m[ --------------------------------------------------- ranked score list --------------------------------------------------- ]\033[0m")
                        pprint({
                            # "err_list": self.err_list,
                            "ranked_score_list": self.outer.ranked_score_list,
                        })

                        from lora_with_llm.try_this.gate import gate_oracle
                        oracle_indices, gate_stats, oracle_configs, configs, proxy_scores = gate_oracle(self.outer.ranked_score_list, self.outer.warm_up_train_x, self.outer.global_oracle_ratio)

                        self.outer.total_candidates += len(configs)
                        self.outer.total_oracle += int(oracle_indices.numel())
                        self.outer.global_oracle_ratio = self.outer.total_oracle / max(1, self.outer.total_candidates)

                        print("\033[1;31m[ --------------------------------------------------- Gate Status --------------------------------------------------- ]\033[0m")
                        pprint({
                            "oracle_indices": oracle_indices, "gate_stats": gate_stats, "oracle_configs": oracle_configs,
                            "total_candidates": self.outer.total_candidates,"total_oracle": self.outer.total_oracle,"global_oracle_ratio": self.outer.global_oracle_ratio,
                        })
                        
                        y_true_list = []
                        x_true_list = []
                        x_true_tensor = None
                        y_true_tensor = None
                        
                        # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                        from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                        rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.outer.proxy_train_x, self.outer.proxy_train_obj, device=DEVICE, save_path="Visualization_3")
                        
                        if len(oracle_configs) > 0:
                            # for i in oracle_indices.tolist():

                            c = x # c has 10 cols
                            
                            print("\033[92m[ok]\033[0m")
                            pprint({
                                "re-calibration_cand": c,
                            })
                            
                            from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                            make_searchspace(c[0], c[1], c[2], c[3],self.seed)

                            from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft

                            ppl, delt_ppl=make_lora_ft('searchspace/configs_single.json', self.model_id)
                            
                            # self.err_list.append(ppl)
                            x_true_list.append(c)
                            y_true_list.append(ppl)

                            self.outer.hard_free()
                        
                            x_true_tensor = torch.stack(x_true_list, dim=0)
                            # (K, 10)
                            y_true_tensor = torch.tensor(y_true_list, **tkwargs).view(-1, 1)   
                            # (K, 1)

                            assert x_true_tensor.shape[0] == y_true_tensor.shape[0], "x and y should have same rows"
                            assert x_true_tensor.shape[1] == 10, "x should have 10 cols"
                            assert y_true_tensor.shape[1] == 1, "y should have 1 cols"

                            # 累积所有“有真值”的点，作为 RankNet 的训练集
                            self.outer.proxy_train_x = torch.cat([self.outer.proxy_train_x, x_true_tensor], dim=0)
                            self.outer.proxy_train_obj = torch.cat([self.outer.proxy_train_obj, y_true_tensor], dim=0)

                            from lora_with_llm.try_this.proxy import fit_proxy_full, save_proxy_ckpt
                            # 用累积到目前为止的所有真值点重新训练 RankNet
                            self.proxy_model, self.proxy_std = fit_proxy_full(
                                self.outer.proxy_train_x,      # (N_all, 10)
                                self.outer.proxy_train_obj,    # (N_all, 1) 只包含 PPL
                                device=DEVICE,
                            )

                            # 可选：偶尔可视化一下 PPL vs proxy 的相关性做 sanity check
                            from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
                            rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, self.outer.proxy_train_x, self.outer.proxy_train_obj, device=DEVICE, save_path="Visualization_3")

                            print(f"rho: {rho}")

                            # 用同一个 ckpt 路径覆盖，这样后面 Gate/Prio 直接加载到的是“最新版本”的 Proxy
                            ckpt_path = "proxy_ckpt.pt"   # 或者你在类里存的 self.proxy_ckpt_path
                            save_proxy_ckpt(ckpt_path, self.proxy_model, self.proxy_std, self.outer.pos2id, self.outer.prec2id)

                        else:
                            # Skip if no re-calibration is triggered
                            pass

                        
                        print("\033[92m[ok]\033[0m")
                        pprint({
                            "proxy_scores": proxy_scores,
                            "lat_list": lat_list,
                        })

                        assert len(proxy_scores) == len(lat_list), "score和lat的数量需要相当"
                        return proxy_scores[0], lat_list[0]

                    else:
                        
                        arch = '{}:{}:{}:{}'.format(x[0], x[1], x[2], x[3]) # 4
                        accelerator = x[4:] # 6
                        
                        ppl, lat = self.rbflex_method(self.image, arch, accelerator)
                        return ppl, lat
                
                f1 = []
                f2 = []

                for q in range(X.shape[0]):
                    x = X[q, :]
                    key = tuple(int(v) for v in x)

                    if key in self.outer.nsga_archive:
                        ppl, lat = self.outer.nsga_archive[key]
                        self.outer.cache_hits += 1      # 先在外层 __init__ 里定义 = 0
                    else:
                        ppl, lat = get_network_score_edp(x)
                        self.outer.cache_miss += 1
                        self.outer.nsga_archive[key] = (ppl, lat)

                    f1.append(ppl)
                    f2.append(lat)

                out["F"] = np.column_stack([f1, f2])

        problem = CustomProblem(
            torch.from_numpy(train_x),
            torch.from_numpy(train_obj),
            bounds_fornorm,
            image,
            self._rbflex_and_dse,
            self.is_score_based,
            outer,
            )

        pop = Population.new("X", train_x)
        static_prob = StaticProblem(problem, F=train_obj)

        pop = Evaluator().eval(static_prob, pop)
        
        # print("___Compare___", pop[0].F, train_obj[0])
        
        algorithm = NSGA2(
            pop_size=len(train_x),
            n_offsprings=batch_size,
            crossover_prob=0.6,
            mutation_prob=0.4,
            eliminate_duplicates=True,
            sampling=pop,
        )
        
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', 2),
            # termination=get_termination("time", "00:00:00"),
            seed=self.outer.seed,
            verbose=True,
            save_history=True,
        )
        
        # callback.plot_hypervolume(res.F)
        val = [e.opt.get("F")[0] for e in res.history]
        generations = np.arange(len(val))
        cycle = [v[0] for v in val]
        acc = [v[1] for v in val]
        
        # # ---------------------- Hypervolume --------------------- #
        # y_tensor = torch.tensor(val)
        # bd = DominatedPartitioning(ref_point=self.ref_point, Y=y_tensor)
        # volume = bd.compute_hypervolume().item()
        # hvs.append(volume)
        # print("### The final hypervolume: ", hvs)
        # # ---------------------- Hypervolume --------------------- #
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        # Plot cycle on the first subplot
        ax1.set_title('Cycle over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Cycle')
        ax1.plot(generations, cycle, color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        # Plot accuracy on the second subplot
        ax2.set_title('Accuracy over Generations')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Accuracy')
        ax2.plot(generations, acc, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        # Adjust layout
        fig.tight_layout(pad=3.0)
        fig.suptitle('Pareto Front Convergence', y=1.02)
        # Save the plot
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, '..', 'COFleX_result', f'{self.Hardware_Arch}_SSS_os', 'plot_saving')
        os.makedirs(target_dir, exist_ok=True)

        logs_dir = os.path.join(target_dir, "Logs")  # 拼出 Logs 子目录路径
        os.makedirs(logs_dir, exist_ok=True)
        
        target_file_path = os.path.join(target_dir, f'pareto_front_related_f{self.current_iteration}.png')
        plt.savefig(target_file_path, dpi=300, bbox_inches='tight')
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(res.F[:, 0], res.F[:, 1])
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_title('f1 vs f2')
        ax.grid(True)
        target_file_path = os.path.join(target_dir, f'pareto_front_2d_{self.current_iteration}.png')
        plt.savefig(target_file_path, dpi=300)
        # plt.show()

        target_file_path = os.path.join(target_dir, f'output_res_f{str(time.time())}.csv')
        df = pd.DataFrame(res.F)
        df.to_csv(target_file_path, index=False)
        
        res_X = res.algorithm.off.get("X")
        candidates = torch.tensor(res_X)
        # print("___This candidates___", self.current_iteration, candidates)

        return candidates
    
    """[Todo] change negative to positive"""
    def plots(self, hsv_list, train_obj):
        fig = plt.figure()
        ax_re = fig.add_subplot(1, 2, 1)
        train_obj = train_obj.cpu().numpy()
        ax_re.scatter(
            train_obj[:, 0], -1 * train_obj[:, 1], alpha=0.8
        )
        ax_re.set_title("AF: {} H-Volume: {}".format(self.acqu_algo, hsv_list[-1]))
        ax_re.set_xlabel("Network Score")
        ax_re.set_ylabel("Cycle")

        ax_2d = fig.add_subplot(1, 2, 2)
        ax_2d.plot(hsv_list)
        ax_2d.set_xlabel("Iteration")
        ax_2d.set_ylabel("H-volume")

        plt.subplots_adjust(wspace=0.4)
        save_dir = "COFleX_result/" + self.Hardware_Arch + "/plot_saving/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + "hypervolume.png", dpi=300)
        # plt.show()
    
    
    def fast_non_dominated_sort(self, P):
        F = defaultdict(list)
        for p in P:
            p.S = []
            p.n = 0
            for q in P:
                if p < q:  # if p dominate q
                    p.S.append(q)  # Add q to the set of solutions dominated by p
                elif q < p:
                    p.n += 1  # Increment the domination counter of p
            if p.n == 0:
                p.rank = 1
                F[1].append(p)
        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.S:
                    q.n = q.n - 1
                    if q.n == 0:
                        q.rank = i + 1
                        Q.append(q)
            i = i + 1
            F[i] = Q
        return F
    
    
    def crowding_distance_assignment(self, L):
        """传进来的参数应该是L = F(i)，类型是List，且 objective 是 ndarray"""
        l = len(L)  # number of solutions in F
        for i in range(l):
            L[i].distance = 0  # initialize distance
        num_objectives = L[0].objective.shape[0]  # number of objectives
        for m in range(num_objectives):
            # Sort using each objective value
            L.sort(key=lambda x: x.objective[m])
            # Boundary points
            L[0].distance = float('inf')
            L[l - 1].distance = float('inf')
            f_max = L[l - 1].objective[m]
            f_min = L[0].objective[m]
            # Avoid division by zero
            if f_max == f_min:
                # print(f"Objective {m}: Max value {f_max} equals Min value {f_min}, skipping.")
                continue
            for i in range(1, l - 1):  # for all other points
                L[i].distance += (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
    
    
    def plot_P(self, P):
        cmap = plt.get_cmap("tab10")  
        colors = [cmap(i) for i in range(len(P))]  
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(P):  
            X = [ind.objective[0] for ind in t]
            Y = [ind.objective[1] for ind in t]
            plt.scatter(X, Y, color=colors[i], alpha=0.5, label=f"Layer {i+1}")  
        plt.xlabel('F1')
        plt.ylabel('F2')
        plt.legend()
        plt.title("Population Distribution Across Layers")
        # plt.show()
        plt.savefig('pareto fronts visualization.png', dpi=300)
        plt.close()
    

    def set_all_seeds(self, sd):
        random.seed(sd); np.random.seed(sd)
        torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _update_memory(self, new_x: torch.Tensor, new_y: torch.Tensor):
        # 1) 追加到历史
        if not torch.is_tensor(self.hist_X):
            self.hist_X = torch.as_tensor(self.hist_X, **tkwargs)
        if not torch.is_tensor(self.hist_Y):
            self.hist_Y = torch.as_tensor(self.hist_Y, **tkwargs)

        def _to_tensor(x, tkwargs):
            if isinstance(x, torch.Tensor):
                # 已经是 Tensor 的情况，只做 dtype/device 对齐
                return x.to(**tkwargs)
            else:
                # list / numpy 的情况，用 as_tensor 更自然一些
                return torch.as_tensor(x, **tkwargs)

        # 然后：
        new_x = _to_tensor(new_x, tkwargs)
        new_y = _to_tensor(new_y, tkwargs)

        old_hist_X = self.hist_X.clone()
        old_hist_Y = self.hist_Y.clone()
        self.hist_X = new_x if self.hist_X is None else torch.cat([self.hist_X, new_x], dim=0)
        self.hist_Y = new_y if self.hist_Y is None else torch.cat([self.hist_Y, new_y], dim=0)

        # 2) 从“全部历史”提取当代 ND
        p_err, p_eng, p_x = self.plot_pareto_acqu_algo(self.iter, self.hist_X.cpu().numpy(), self.hist_Y.cpu().numpy(), old_hist_X.cpu().numpy(), old_hist_Y.cpu().numpy(), new_x.cpu().numpy(), new_y.cpu().numpy(), id=0)
        train_x = torch.tensor(p_x, **tkwargs)
        train_obj = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)

        ndX = train_x
        ndY = train_obj

        # 3) 与旧 archive 合并后再次取 ND，得到“跨轮最优”
        if self.arc_X is None:
            self.arc_X, self.arc_Y = ndX, ndY
        else:
            
            old_arc_X = self.arc_X.clone()
            old_arc_Y = self.arc_Y.clone()
            
            Xc = torch.cat([self.arc_X, ndX], dim=0)
            Yc = torch.cat([self.arc_Y, ndY], dim=0)
            # nd = is_non_dominated(Yc)
            
            p_err, p_eng, p_x = self.plot_pareto_acqu_algo(self.iter, Xc.cpu().numpy(), Yc.cpu().numpy(), old_arc_X.cpu().numpy(), old_arc_Y.cpu().numpy(), ndX.cpu().numpy(), ndY.cpu().numpy(), id=1)
            train_x = torch.tensor(p_x, **tkwargs)
            train_obj = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)

            self.arc_X, self.arc_Y = train_x, train_obj

        return self.arc_X, self.arc_Y
    
    # def _dedup(self, X, Y, bounds, eps=0.01):
    #     # 到 [0,1]^d
    #     Xn = normalize(X.to(**tkwargs), bounds.to(**tkwargs))
    #     # 网格化去重：把 [0,1] 分成 1/eps 的格
    #     keys = (Xn / eps).floor()
    #     _, uniq_idx = torch.unique(keys, dim=0, return_index=True)
    #     return X[uniq_idx], Y[uniq_idx]

    def _dedup(self, X, Y, bounds, eps=0.01):
        # 统一设备/精度
        X = X.to(**tkwargs); Y = Y.to(**tkwargs); B = bounds.to(**tkwargs)

        # 归一化到 [0,1]^d 并做 1/eps 网格键
        Xn = normalize(X, B)
        keys = (Xn / eps).floor()                      # (N, d) float/long 均可

        # unique 不支持 return_index，用 return_inverse 自己恢复“首个索引”
        _, inv = torch.unique(keys, dim=0, return_inverse=True)   # inv: (N,)
        order = torch.argsort(inv)                                # 把同一组放一起
        inv_sorted = inv[order]
        # 每个分组的第一个为“首出现”
        first_mask = torch.ones_like(inv_sorted, dtype=torch.bool)
        first_mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
        uniq_idx = order[first_mask]                               # (K,)

        return X[uniq_idx], Y[uniq_idx]
    
    def _cover_subsample(self, X, k, bounds):
        if X.shape[0] <= k: return torch.arange(X.shape[0], device=X.device)
        Xn = normalize(X.to(**tkwargs), bounds.to(**tkwargs))
        # 随机起点
        idx0 = torch.randint(Xn.shape[0], (1,), device=X.device)
        selected = [idx0.item()]
        dist_min = torch.cdist(Xn[idx0], Xn).squeeze(0)  # 到已选集合的最小距离
        for _ in range(k-1):
            j = torch.argmax(dist_min).item()
            selected.append(j)
            dist_min = torch.minimum(dist_min, torch.cdist(Xn[j:j+1], Xn).squeeze(0))
        return torch.tensor(selected, device=X.device, dtype=torch.long)

    def _explore_around_frontier(self,
                                n: int,
                                bounds: torch.Tensor,
                                frontier_X: torch.Tensor,
                                min_dist: float = 0.02,
                                radius: float = 0.05,
                                max_draw: int = 4096*2):
        """
        在已有 Pareto front (frontier_X) 的邻域里，探索 n 个新点：
        - 在 [0,1]^D 中：先把 frontier_X 归一化，作为 anchor
        - 每个候选点 = 随机选一个 anchor + 小扰动（半径 radius）
        - 同时保证与历史点和已选点的距离 >= min_dist
        """

        device = bounds.device
        dtype = bounds.dtype
        D = bounds.shape[1]

        # 1) 归一化 Pareto front 到 [0,1]^D
        front_n = normalize(frontier_X.to(device=device, dtype=dtype), bounds)   # [M, D]

        # 2) 归一化历史点（可选：这里你也可以只用 hist 中比较好的点）
        if getattr(self, "hist_X", None) is not None and self.hist_X.numel() > 0:
            hist_n = normalize(self.hist_X.to(device=device, dtype=dtype), bounds)
        else:
            hist_n = None

        eng = SobolEngine(dimension=D,
                        scramble=True,
                        seed=getattr(self, "seed", None))

        out_unit = []
        draws = 0

        while len(out_unit) < n and draws < max_draw:
            # 2.1 随机选一个 Pareto front 上的 anchor
            idx = torch.randint(0, front_n.shape[0], (1,), device=device)
            anchor = front_n[idx]   # [1, D]

            # 2.2 在一个超球/超箱内加扰动，控制 "radius"
            # Sobol 出一个 [0,1]^D 点，映射到 [-radius, radius]^D
            delta = eng.draw(1).to(device=device, dtype=dtype)
            delta = (delta * 2.0 - 1.0) * radius

            cand = torch.clamp(anchor + delta, 0.0, 1.0)   # 仍然在 [0,1]^D
            draws += 1

            # 2.3 跟历史点的距离约束：不要太贴脸，以免重样
            if hist_n is not None and hist_n.numel() > 0:
                d_hist = torch.cdist(cand, hist_n).min()
                if d_hist < min_dist:
                    continue

            # 2.4 跟已选探索点的距离约束
            if out_unit:
                Sel = torch.cat(out_unit, dim=0)   # [m, D]
                d_out = torch.cdist(cand, Sel).min()
                if d_out < min_dist:
                    continue

            out_unit.append(cand)

        if not out_unit:
            # 兜底：没采到，就退回你原来的全局探索
            return self._novel_samples(n, bounds, min_dist=min_dist)

        print(f"[EX] generated {len(out_unit)} / {n}, draws={draws}")

        U = torch.cat(out_unit, dim=0)[:n]           # [n, D] in [0,1]
        X_raw = unnormalize(U, bounds)               # 回原始域
        return X_raw

    def _novel_samples(self,
        n: int,
        bounds: torch.Tensor,
        min_dist: float = 0.05,
        max_draw: int = 4096,
        batch: int = 512) -> torch.Tensor:
        """
        目的：在尚未覆盖的区域生成 n 个探索性候选（返回原始数值域）。
        - min_dist: 在 [0,1]^d 归一化坐标系下与“历史点+已选新点”的最小欧氏距离阈值。
        - max_draw: 最多尝试的候选数量上限（防止死循环）。
        - batch: 每次从 Sobol 引擎拉取的候选批量。
        """
        D = bounds.shape[1]
        device, dtype = bounds.device, bounds.dtype

        # 历史点（若有）→ 归一化
        if getattr(self, "hist_X", None) is not None and self.hist_X.numel() > 0:
            hist_n = normalize(self.hist_X.to(device=device, dtype=dtype), bounds)
        else:
            hist_n = None

        eng = SobolEngine(dimension=D, scramble=True,
                        seed=getattr(self, "seed", None))
        out_unit = []            # 已选的新点（单位化空间）
        drawn = 0

        while len(out_unit) < n and drawn < max_draw:
            B = min(batch, max_draw - drawn)
            U = eng.draw(B).to(device=device, dtype=dtype)      # [B, D] ∈ [0,1]
            drawn += B

            # 先与历史点做距离过滤
            if hist_n is not None and hist_n.numel() > 0:
                d = torch.cdist(U, hist_n)                      # [B, |H|]
                keep = d.min(dim=1).values >= min_dist
                U = U[keep]
                if U.numel() == 0:
                    continue

            # 再与“已选新点”做两两最小距离约束（贪心插入，保证 pairwise ≥ min_dist）
            if out_unit:
                Sel = torch.cat(out_unit, dim=0)               # [m, D]
                # 逐条插入，保证与已选集合的最小距离
                for row in U:
                    d2 = torch.cdist(row.unsqueeze(0), Sel).min()
                    if d2 >= min_dist:
                        out_unit.append(row.unsqueeze(0))
                    if len(out_unit) >= n:
                        break
            else:
                # 如果还没有已选点，先收一批，再逐条加严
                if U.shape[0] > 0:
                    out_unit.append(U[0:1])
                    for row in U[1:]:
                        d2 = torch.cdist(row.unsqueeze(0), out_unit[0]).min()
                        if d2 >= min_dist:
                            out_unit.append(row.unsqueeze(0))
                        if len(out_unit) >= n:
                            break

        if not out_unit:
            # 兜底：直接返 n 个 Sobol 点（未做距离过滤）
            U = eng.draw(n).to(device=device, dtype=dtype)
            return unnormalize(U, bounds)

        U = torch.cat(out_unit, dim=0)[:n]                     # [n, D] in [0,1]
        X_raw = unnormalize(U, bounds)                         # 回原始域
        return X_raw
    
    def _build_training_pool(self, bounds, B_total=1024, B_explore=256, eps=0.01):
        assert self.hist_X is not None
        
        HX, HY = self._dedup(self.hist_X, self.hist_Y, bounds, eps=eps)

        AX, AY = (torch.empty(0, HX.shape[1], **tkwargs), torch.empty(0, HY.shape[1], **tkwargs)) \
                if self.arc_X is None else (self.arc_X, self.arc_Y)

        # 2) 从“历史被支配集”做覆盖采样
        # 找到不在 Archive 中的 those dominated points
        p_err, p_eng, p_x = self.plot_pareto_acqu_algo(self.iter, HX.cpu().numpy(), HY.cpu().numpy(), id=2)
        
        train_x = torch.tensor(p_x, **tkwargs)
        train_obj = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)

        DX, DY = train_x, train_obj

        budget_left = max(0, B_total - AX.shape[0])
        B_cover = max(0, budget_left - B_explore)

        if DX.shape[0] > 0 and B_cover > 0:
            sel = self._cover_subsample(DX, min(B_cover, DX.shape[0]), bounds)
            CX, CY = DX[sel], DY[sel]
        else:
            CX, CY = torch.empty(0, HX.shape[1], **tkwargs), torch.empty(0, HY.shape[1], **tkwargs)

        # 3) 在“空白区”投放探索点（只加入 X；其 Y 由下一步评估得到）
        # EX = self._novel_samples(B_explore, bounds, min_dist=0.05)
        if self.arc_X is not None and self.arc_X.numel() > 0:
            EX = self._explore_around_frontier(n=B_explore,
                                        bounds=self.bounds,
                                        frontier_X=self.arc_X,
                                        min_dist=0.14,
                                        radius=0.20)
        # else:
        #     EX = self._novel_samples(B_explore, self.bounds, min_dist=0.05)
        
        # 拼装现有可训练数据（X_train, Y_train）
        X_train = torch.cat([AX, CX], dim=0)
        Y_train = torch.cat([AY, CY], dim=0)

        return X_train, Y_train, EX
    
    def _to_2d_cpu_ndarray(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().to('cpu').numpy()
        else:
            x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def _save_csv(self, arr, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(arr)
        # 无表头、无索引，纯数据
        df.to_csv(path, index=False, header=False)

    def save_iter_snapshots(self, root_dir="snapshots"):
        it = int(self.iter)  # 当前代号
        root = Path(root_dir)

        items = [
            ("hist/X",     self.hist_X),
            ("hist/Y",     self.hist_Y),
            ("archive/X",  self.arc_X),
            ("archive/Y",  self.arc_Y),
            ("explore/X",  self.EX),   # EX 只有 X
        ]

        for sub, obj in items:
            if obj is None:
                continue
            arr = self._to_2d_cpu_ndarray(obj)
            self._save_csv(arr, root / sub / f"{it}.csv")

    def run(self):
        print('+++++++++++ Optimization +++++++++++')
        print('==> Reproducibility..')

        self.set_all_seeds(self.seed)

        self.cleanup_proxy_ckpt()

        LOAD_PATH= Path(__file__).resolve().parents[2]   # …/H2-LLM-ISCA-2025
 
        if self.model_id == 'meta-llama/Llama-3.2-3B':
            # load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_01_22_21_38" # meta-llama/Llama-3.2-3B
            load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_02_01_19_10" # meta-llama/Llama-3.2-3B
            # load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_xxx_xxx" # meta-llama/Llama-3.2-3B
        
        elif self.model_id == 'Qwen/Qwen2.5-1.5B':
            load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_xxx_xxx" # Qwen/Qwen2.5-1.5B
        
        elif self.model_id == 'meta-llama/Llama-3.1-8B':
            # load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_xxx_xxx" # meta-llama/Llama-3.1-8B
            load_path = LOAD_PATH / "hw_nas" / "COflex_saving" / "_INIT_DATA_SIZE100_TIME_02_02_05_44" # meta-llama/Llama-3.1-8B
        # print(load_path)
        
        if os.path.exists(load_path):

            train_x_path = f"{load_path}/train_input.csv"

            train_obj_path = f"{load_path}/train_output.csv"
            
            if os.path.exists(train_x_path) and os.path.exists(train_obj_path):
                try:
                    train_x = pd.read_csv(train_x_path, header=None).iloc[:, :10].values
                    train_obj = pd.read_csv(train_obj_path, header=None).iloc[:, :2].values

                    train_x = torch.tensor(train_x, **tkwargs)
                    train_obj = torch.tensor(train_obj, **tkwargs)

                    print("Data loaded successfully:")
                    print(f"train_x shape: {train_x.shape}, train_obj shape: {train_obj.shape}")

                except Exception as e:
                    print(f"Error loading data: {e}")
            else:
                pass
        
        else:
            print("Required files not found at the specified paths.")
            print("-- No file found at the specified path, save another --")

            train_x, train_obj = self._generate_initial_data(self.image, n=self.n_init_size)

            save_dir = Path("COflex_saving") / f"_INIT_DATA_SIZE{self.n_init_size}_TIME_{datetime.now():%m_%d_%H_%M}"
            save_dir.mkdir(parents=True, exist_ok=True)
            try:

                x = train_x.detach().cpu().numpy()
                y = train_obj.detach().cpu().numpy()

                if x.ndim == 1: x = x[:, None]
                if y.ndim == 1: y = y[:, None]

                pd.DataFrame(x).to_csv(save_dir / "train_input.csv",  index=False, header=False)
                pd.DataFrame(y).to_csv(save_dir / "train_output.csv", index=False, header=False)
            except Exception as e:
                print(f"Error saving files: {e}")

        if self.is_score_based:
            HERE = Path(__file__).resolve()
            ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
            sys.path.insert(0, str(ROOT_LORA))

            self.proxy_train_x = train_x[:, :4].clone()     # (N0, 4)
            self.proxy_train_obj = train_obj[:, :1].clone() # (N0, 1)
            
            # /home/myh/H2-LLM-ISCA-2025/lora_with_llm/try_this/proxy.py
            
            from lora_with_llm.try_this.proxy import fit_proxy_full, save_proxy_ckpt
            self.proxy_model, self.proxy_std = fit_proxy_full(train_x, train_obj, device=DEVICE)

            from lora_with_llm.try_this.proxy import visualize_score_vs_ppl
            rho = visualize_score_vs_ppl(self.proxy_model, self.proxy_std, train_x, train_obj, device=DEVICE, save_path="Visualization_4")

            print(f"rho: {rho}")

            save_proxy_ckpt("proxy_ckpt.pt", self.proxy_model, self.proxy_std, self.pos2id, self.prec2id)

            from lora_with_llm.try_this.proxy import predict_from_Tensor
            
            # ranked: <class 'list'>
            ckpt_path = "proxy_ckpt.pt"
            train_x, train_obj = predict_from_Tensor(ckpt_path, train_x, train_obj, device=DEVICE)

            self.warm_up_train_x = train_x
            self.warm_up_train_obj = train_obj

        self.cost_time = time.time() - self.start_time
        
        self.OPT_VS_TIME_REC = torch.cat((train_obj, torch.full((train_obj.shape[0], 1), self.cost_time, **tkwargs)), dim=1)
        assert self.OPT_VS_TIME_REC.shape[1] == 3, "OPT_VS_TIME_REC should have 3 columns"
        
        self.TOTAL_TRAIN_X_REC = train_x
        self.TOTAL_TRAIN_OBJ_REC = train_obj
        self.start_time = time.time()

        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "train_x": train_x,
        #     "train_obj": train_obj,
        # })

        train_x_err = train_x[:, :4].cpu().numpy() # θ
        train_x_eng = train_x[:, 4:].cpu().numpy() # Γ

        train_obj_err = train_obj[:, 0]
        train_obj_eng = train_obj[:, 1]
        
        train_err_recording = train_x_err.copy() # Dperf
        train_eng_recording = train_x_eng.copy() # Deng
        
        IntPar_recording = []
        self.observation_theta.append(train_obj_err) # O(θ)
        self.observation_gamma.append(train_obj_eng) # O(Γ)

        assert train_err_recording.shape[0] == train_eng_recording.shape[0], "Initial ERR/ENG samples must align"

        train_x = torch.tensor(np.hstack((train_err_recording, train_eng_recording)), **tkwargs)
        train_obj = torch.cat((torch.cat(self.observation_theta, dim=0).unsqueeze(1), torch.cat(self.observation_gamma, dim=0).unsqueeze(1)), dim=1)

        def debug_check_unit_cube(X, bounds, name):
            X = X.detach().to(torch.double)
            bounds = bounds.detach().to(X)
            lo, hi = bounds[0], bounds[1]
            assert lo.shape == hi.shape == X.shape[-1:], f"{name}: bounds shape mismatch"
            assert torch.all(hi > lo), f"{name}: some upper<=lower"

            Xn = (X - lo) / (hi - lo)
            mn, mx = Xn.min().item(), Xn.max().item()
            bad_cols = torch.nonzero((Xn < -1e-8).any(dim=0) | (Xn > 1+1e-8).any(dim=0), as_tuple=False).flatten()
            print(f"[{name}] Xn range = [{mn:.6f}, {mx:.6f}], bad_cols = {bad_cols.tolist()}")

        # debug_check_unit_cube(train_x,         self.bounds_fornorm, "full")
        # debug_check_unit_cube(train_x[:, :3],  self.bounds_err,     "err")
        # debug_check_unit_cube(train_x[:, 3:],  self.bounds_eng,     "eng")
        
        # print("\033[92m[ok]\033[0m")
        # pprint({
        #     "bounds_fornorm": self.bounds_fornorm,
        #     "bounds_err": self.bounds_err,
        #     "bounds_eng": self.bounds_eng,
        # })

        def report_bad_values(X: torch.Tensor, name="train_x", limit=100):
            """打印张量中 NaN/Inf 的位置与统计；limit 控制示例坐标数量。"""
            assert X.ndim == 2, f"{name} should be 2D, got shape={tuple(X.shape)}"
            mask_nan = torch.isnan(X)
            mask_inf = ~torch.isfinite(X) & ~mask_nan

            n_nan = int(mask_nan.sum().item())
            n_inf = int(mask_inf.sum().item())

            print(f"[{name}] shape={tuple(X.shape)} dtype={X.dtype} device={X.device}")
            print(f"  NaN count = {n_nan},  +/-Inf count = {n_inf}")

            if n_nan:
                rows_nan = torch.nonzero(mask_nan.any(dim=1), as_tuple=False).squeeze(1)
                cols_nan = torch.nonzero(mask_nan.any(dim=0), as_tuple=False).squeeze(1)
                print(f"  rows containing NaN ({rows_nan.numel()}): {rows_nan[:limit].tolist()}"
                    + (" ..." if rows_nan.numel() > limit else ""))
                print(f"  cols containing NaN ({cols_nan.numel()}): {cols_nan.tolist()}")

                # 每列 NaN 个数
                col_counts = mask_nan.sum(dim=0).cpu().tolist()
                for j in cols_nan.cpu().tolist():
                    print(f"    col {j}: {int(col_counts[j])} NaNs")

                # 打印若干具体坐标
                coords = torch.nonzero(mask_nan, as_tuple=False)
                for k in range(min(limit, coords.size(0))):
                    i, j = coords[k].tolist()
                    print(f"    NaN at (row={i}, col={j})")

            if n_inf:
                rows_inf = torch.nonzero(mask_inf.any(dim=1), as_tuple=False).squeeze(1)
                cols_inf = torch.nonzero(mask_inf.any(dim=0), as_tuple=False).squeeze(1)
                print(f"  rows containing Inf ({rows_inf.numel()}): {rows_inf[:limit].tolist()}"
                    + (" ..." if rows_inf.numel() > limit else ""))
                print(f"  cols containing Inf ({cols_inf.numel()}): {cols_inf.tolist()}")

            # 也可作为硬性检查
            assert torch.isfinite(X).all(), f"{name} contains NaN/Inf — see logs above."
        report_bad_values(train_x)

        mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
        mll_err, model_err = self._initialize_model(train_x[:, :4], train_obj[:, :1], self.bounds_err)
        mll_eng, model_eng = self._initialize_model(train_x[:, 4:], train_obj[:, 1:], self.bounds_eng)
        
        hvs = []
        # Reference points
        min_values, _ = torch.min(train_obj, dim=0)
        if self.Hardware_Arch == "DL2":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.ENERGY_IDX] = min_values[self.ENERGY_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        
        elif self.Hardware_Arch == "ScaleSim":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        
        elif self.Hardware_Arch == "DeFiNES":
            self.ref_point[self.ERROR_IDX] = min_values[self.ERROR_IDX]
            self.ref_point[self.EDP_IDX] = min_values[self.EDP_IDX]

        elif self.Hardware_Arch == "H^2":
            pass

        l = torch.tensor([1.0, 0.1], device=train_obj.device)
        r = torch.tensor([10.0, 1.0], device=train_obj.device)

        Y_norm_min = (train_obj - l) / (r - l)
        Y_norm_min = torch.clamp(Y_norm_min, 0.0, 1.0)

        ref_norm_min = (self.ref_point - l) / (r - l)
        ref_norm_min = torch.clamp(ref_norm_min, 0.0, 1.0)
        
        bd = DominatedPartitioning(ref_point=-ref_norm_min, Y=-Y_norm_min)
        volume = bd.compute_hypervolume().item()
        hvs.append(volume)
        
        # print()
        print("[init] Hypervolume: {}".format(self.N_BATCH, hvs[-1]))
        for iteration in range(1, self.N_BATCH + 1):

            if time.time() - self.start_time > 2 * 8 * 60 * 60:                
                self.time_used_list.append(time.time())

                # if self.is_score_based:
                #     if self.arc_X is None or self.arc_Y is None or self.arc_X.numel() == 0:
                #         print("Archive is empty; skip score->truth conversion.")
                #     else:
                #         print(" -- Convert score to true ppl value -- ")
                #         ppl_list = []
                #         archiv_train_x = self.arc_X

                #         latency_tensor = self.arc_Y[:, 1].view(-1, 1)

                #         # HERE = Path(__file__).resolve()
                #         # ROOT_LORA = HERE.parents[2]  # H2-LLM-ISCA-2025（Simulator -> hw_nas -> ROOT）
                #         # if str(ROOT_LORA) not in sys.path:
                #         #     sys.path.insert(0, str(ROOT_LORA))

                #         from lora_with_llm.try_this.sweet_spot_gen_searchspace import make_searchspace
                #         from lora_with_llm.try_this.esm.sweet_spot_run_lora_screener_GA import make_lora_ft

                #         for config in self.arc_X[:, :4].tolist():
                #             make_searchspace(config[0], config[1], config[2], config[3], self.seed)
                #             ppl, _ = make_lora_ft('searchspace/configs_single.json', self.model_id)
                #             ppl_list.append(ppl)
                #             self.hard_free()

                #         ppl_tensor = torch.tensor(ppl_list, **tkwargs).view(-1, 1)
                #         archiv_train_obj = torch.cat(
                #             (ppl_tensor, latency_tensor.to(**tkwargs)), dim=1
                #         )

                #         self.archiv_train_x = archiv_train_x
                #         self.archiv_train_obj = archiv_train_obj
                break
            
            print(f"\033[1;31m[ Time used: --------------------------------------------------- {time.time() - self.start_time} --------------------------------------------------- ]\033[0m")
            self.time_used_list.append(time.time())
            
            self.hard_free()
            
            self.iter = iteration

            if self.acqu_algo == "nsga" or self.acqu_algo == "nsga_ppl":
                print("-- Skip ---")
            else:
                fit_gpytorch_mll(mll)
            
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES])) # define the qEI and qNEI acquisition modules using a QMC sampler
                
            if self.acqu_algo == "Coflex" or self.acqu_algo == "Coflex_ppl":
                assert train_err_recording.shape[0] == train_eng_recording.shape[0], "ERR/ENG history length mismatch"
                X = np.hstack((train_err_recording, train_eng_recording)) #(Dperf, fperf)


                def _as_1d_list(tlist):
                    return [torch.as_tensor(t, **tkwargs).flatten() for t in tlist]
                
                y = torch.cat(
                    (torch.cat(_as_1d_list(self.observation_theta), dim=0).unsqueeze(1), 
                     torch.cat(_as_1d_list(self.observation_gamma), dim=0).unsqueeze(1)), 
                dim=1).cpu().numpy() # (Deng, feng)
                
                self.hist_X = X if self.hist_X is None else torch.cat([self.hist_X, torch.tensor(X, **tkwargs)], dim=0)
                self.hist_Y = y if self.hist_Y is None else torch.cat([self.hist_Y, torch.tensor(y, **tkwargs)], dim=0)

                X_err = X[:, :4]
                X_eng = X[:, 4:]
                y_err = y[:, 0:1]
                y_eng = y[:, 1:2]
                
                fit_gpytorch_mll(mll_eng)
                fit_gpytorch_mll(mll_err)

                print("\033[92m[ok]\033[0m")
                pprint({
                    "iteration": iteration,
                })

                self.status = "ERR"
                theta_next, o_theta_next = self._get_new_data(self.image, self.acqu_algo, model_err, X_err, y_err, sampler) # θn+1, O(θn+1) -> ERR
                self.status = "ENG"
                gamma_next, o_gamma_next = self._get_new_data(self.image, self.acqu_algo, model_eng, X_eng, y_eng, sampler) # Γn+1, O(Γn+1) -> EDP
                
                print("\033[92m[ok]\033[0m")
                # pprint({
                #     "theta_next": theta_next,
                #     "o_theta_next": o_theta_next,
                #     "gamma_next": gamma_next,
                #     "o_gamma_next": o_gamma_next,
                # })

                self.cost_time = time.time() - self.start_time
                theta = torch.as_tensor(o_theta_next).to(**tkwargs).flatten().unsqueeze(1)  # (N,1)
                gamma = torch.as_tensor(o_gamma_next).to(**tkwargs).flatten().unsqueeze(1)  # (N,1)

                row_obj = torch.cat((theta, gamma), dim=1)                     # (N,2)
                cost  = torch.full((row_obj.size(0), 1), self.cost_time, **tkwargs)  # (N,1)
                row   = torch.cat((row_obj, cost), dim=1)                      # (N,3)

                row_x = torch.cat((theta_next, gamma_next), dim=1)

                self.OPT_VS_TIME_REC = torch.cat((self.OPT_VS_TIME_REC, row), dim=0)

                if not hasattr(self, "TOTAL_TRAIN_OBJ_REC") or self.TOTAL_TRAIN_OBJ_REC is None:
                    self.TOTAL_TRAIN_OBJ_REC = torch.empty((0, 2), **tkwargs)

                self.TOTAL_TRAIN_OBJ_REC = torch.cat((self.TOTAL_TRAIN_OBJ_REC, row_obj), dim=0)

                if not hasattr(self, "TOTAL_TRAIN_X_REC") or self.TOTAL_TRAIN_X_REC is None:
                    self.TOTAL_TRAIN_X_REC = torch.empty((0, 10), **tkwargs)

                self.TOTAL_TRAIN_X_REC = torch.cat((self.TOTAL_TRAIN_X_REC, row_x), dim=0)
                
                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "self.TOTAL_TRAIN_X_REC": self.TOTAL_TRAIN_X_REC,
                # })
                
                train_err_recording = np.vstack([train_err_recording, theta_next.cpu().numpy()])
                train_eng_recording = np.vstack([train_eng_recording, gamma_next.cpu().numpy()])
                
                self.observation_theta.append(o_theta_next) # O(θn+1)
                self.observation_gamma.append(o_gamma_next) # O(Γn+1)
                
                # for item in self.observation_theta:
                #     print(item, item.shape)              
                # for item in self.observation_gamma:
                #     print(item, item.shape)

                def _as_1d_list(tlist):
                    return [torch.as_tensor(t, **tkwargs).flatten() for t in tlist]

                y_err = torch.cat(_as_1d_list(self.observation_theta), dim=0).cpu().numpy()
                y_eng = torch.cat(_as_1d_list(self.observation_gamma), dim=0).cpu().numpy()

                assert y_err.shape[0] == y_eng.shape[0]

                assert train_err_recording.shape[0] == train_eng_recording.shape[0], "ERR/ENG history length mismatch"
                X = np.hstack((train_err_recording, train_eng_recording))
                y = np.stack((y_err, y_eng), axis=1)
                
                # p_err, p_eng, p_x = self.pareto_check(iteration, X, y)

                self.beta = torch.tensor(X, **tkwargs)
                # ---------------------- Level2 --------------------- # 
                # print(self.beta.shape) # βn
                
                self.IntPar.append(self.beta) # DIntPar + βn
                # On_beta = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)# On(β)
                On_beta = torch.tensor(y, **tkwargs)
                # print(On_beta.shape)
                
                assert self.beta.shape[1] == 10, "self.beta should have 10 columns"
                assert On_beta.shape[1] == 2, "On_beta should have 2 columns"
                
                observation_norm = (np.array(On_beta.cpu().numpy()) - np.array(On_beta.cpu().numpy()).min(axis=0)) / \
                            (np.array(On_beta.cpu().numpy()).max(axis=0) - np.array(On_beta.cpu().numpy()).min(axis=0) + 1e-8)            
                l1_norms = np.sum(observation_norm, axis=1)
                
                IntPar = self.beta[np.argsort(l1_norms)] # train_x
                self.DIntPar = On_beta # train_ob
                
                fit_gpytorch_mll(mll)
                
                self.status = "NOM"
                new_X_tensor, new_y_tensor = self._get_new_data(self.image, self.acqu_algo, model, IntPar, self.DIntPar, sampler) # βn+1, O(βn+1)
                # ***
                self.cost_time = time.time() - self.start_time

                self.OPT_VS_TIME_REC = torch.cat([self.OPT_VS_TIME_REC, torch.cat((new_y_tensor, torch.full((new_y_tensor.shape[0], 1), self.cost_time, **tkwargs)), dim=1)])
                self.TOTAL_TRAIN_X_REC = torch.cat([self.TOTAL_TRAIN_X_REC, new_X_tensor])
                self.TOTAL_TRAIN_OBJ_REC = torch.cat([self.TOTAL_TRAIN_OBJ_REC, new_y_tensor])
                
                train_err_recording = np.vstack([train_err_recording, new_X_tensor[:, :4].cpu().numpy()])
                train_eng_recording = np.vstack([train_eng_recording, new_X_tensor[:, 4:].cpu().numpy()])
                
                self.observation_theta.append(new_y_tensor[:,0])
                self.observation_gamma.append(new_y_tensor[:,1])

                assert train_err_recording.shape[0] == train_eng_recording.shape[0], "ERR/ENG history length mismatch"
                train_x = torch.tensor(np.hstack((train_err_recording, train_eng_recording)), **tkwargs)

                def _as_1d_list(tlist):
                    return [torch.as_tensor(t, **tkwargs).reshape(-1) for t in tlist]

                train_obj = torch.cat(
                    (torch.cat(_as_1d_list(self.observation_theta), dim=0).unsqueeze(1), 
                     torch.cat(_as_1d_list(self.observation_gamma), dim=0).unsqueeze(1)), 
                dim=1)

                train_x, train_obj = self._update_memory(train_x, train_obj)

                train_x, train_obj, EX = self._build_training_pool(self.bounds, B_total=1024, B_explore=256)
                self.EX = EX

                assert train_x.shape[1] == 10, "train_x should have 10 columns"
                assert train_obj.shape[1] == 2, "train_obj should have 2 columns"
                assert train_x.shape[0] == train_obj.shape[0], "train_x and train_obj should have the same number of rows"
                assert train_x.ndim == 2, "train_x should be a 2D tensor"
                assert train_obj.ndim == 2, "train_obj should be a 2D tensor"

                mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
                mll_err, model_err = self._initialize_model(train_x[:, :4], train_obj[:, 0].unsqueeze(1), self.bounds_err)
                mll_eng, model_eng = self._initialize_model(train_x[:, 4:], train_obj[:, 1].unsqueeze(1), self.bounds_eng)
                
                # print("\033[92m[ok]\033[0m")
                # pprint({
                #     "self.ref_point": self.ref_point,
                # })

                l = torch.tensor([1.0, 0.1], device=train_obj.device)
                r = torch.tensor([10.0, 1.0], device=train_obj.device)

                Y_norm_min = (train_obj - l) / (r - l)
                Y_norm_min = torch.clamp(Y_norm_min, 0.0, 1.0)

                ref_norm_min = (self.ref_point - l) / (r - l)
                ref_norm_min = torch.clamp(ref_norm_min, 0.0, 1.0)
                
                bd = DominatedPartitioning(ref_point=-ref_norm_min, Y=-Y_norm_min)
                volume = bd.compute_hypervolume().item()
                hvs.append(volume)

            else:
                new_x, new_obj = self._get_new_data(self.image, self.acqu_algo, model, train_x, train_obj, sampler)

                old_train_x = train_x.clone()
                old_train_obj = train_obj.clone()
                
                train_x = torch.cat([train_x.to(**tkwargs), new_x.to(**tkwargs)])
                # ↑ self.hist_X
                self.hist_X = train_x

                train_obj = torch.cat([train_obj.to(**tkwargs), new_obj.to(**tkwargs)])
                # ↑ self.hist_Y
                self.hist_Y = train_obj

                p_err, p_eng, p_x = self.plot_pareto_acqu_algo(self.iter, train_x.cpu().numpy(), train_obj.cpu().numpy(), old_train_x.cpu().numpy(), old_train_obj.cpu().numpy(), new_x.cpu().numpy(), new_obj.cpu().numpy(), id=3)

                self.arc_X, self.arc_Y = torch.tensor(p_x, **tkwargs), torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)

                bd = DominatedPartitioning(ref_point=-self.ref_point, Y=-train_obj)
                volume = bd.compute_hypervolume().item()
                hvs.append(volume)

            if not (self.acqu_algo == "random" or self.acqu_algo == "random_ppl" or self.acqu_algo == "SWoptOnlyrandom_ppl" or self.acqu_algo == "HWoptOnlyrandom_ppl" or self.acqu_algo == "Coflex" or self.acqu_algo == "Coflex_ppl" or self.acqu_algo == "nsga" or self.acqu_algo == "nsga_ppl"):
                mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)

            safe_model_id = self.model_id.replace("/", "_")

            # 全局统一，局外不同
            if self.ts is None:
                self.ts = datetime.now().strftime("_%m_%d_%H_%M_")
            
            self.save_iter_snapshots(root_dir=f"./snapshots/{self.acqu_algo}_{safe_model_id}_{self.Hardware_Arch}_{self.seed}_{self.ts}/")

            print("iteration [{}/{}] Hypervolume: {}".format(iteration, self.N_BATCH, hvs[-1]))
        
        # Show top-5 result
        # self.delete_all_folders('inputs/WL/Meta_prototype_DF')
        print('+++++++++++ Result +++++++++++')
        torch.set_printoptions(precision=2, linewidth=100)
        print("H-Volume: ", hvs[-1])
        
        if self.Hardware_Arch == "DL2":
            if self.benchmark == "sss":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i,0]),int(optimal_x[i,1]),int(optimal_x[i,2]),int(optimal_x[i,3]),int(optimal_x[i,4]))
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tEnergy (FAKE): ", -1*optimal_obj[i,1].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,2].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tEnergy (FAKE): ", -1*optimal_obj[i,1].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,2].item()))
        
        elif self.Hardware_Arch == "ScaleSim":
            if self.benchmark == "sss":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i,0]),int(optimal_x[i,1]),int(optimal_x[i,2]),int(optimal_x[i,3]),int(optimal_x[i,4]))
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tHardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,1].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,1].item()))
        
        elif self.Hardware_Arch == "H^2":
            optimal_x = train_x
            optimal_obj = train_obj
            # print("Backbone Design Space: NATS-Bench-SSS")
            for i in range(len(optimal_x)):
                arch = '{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]))

                acce = {"X1": optimal_x[i, 4].item(), "X2": optimal_x[i, 5].item(), "X3": optimal_x[i, 6].item(),
                        "X4": optimal_x[i, 7].item(), "X5": optimal_x[i, 8].item()}
                
                print("* Candidate[{}]".format(i + 1))
                print("\tBackbone architecture: {}".format(arch))
                print("\tDeFiNES Hardware: ", acce)
                print("\t----------------------------------")
                print("\tErr ", (optimal_obj[i, 0].item()), "%")
                print("\tEDP: ",(optimal_obj[i, 1].item()), "µJ*s")

        elif self.Hardware_Arch == "DeFiNES":
            if self.benchmark == "sss":
                optimal_x = train_x
                optimal_obj = train_obj
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(len(optimal_x)):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]),
                                                   int(optimal_x[i, 3]), int(optimal_x[i, 4]))
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDeFiNES Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tErr ", (optimal_obj[i, 0].item()), "%")
                    print("\tEDP: ",(optimal_obj[i, 1].item()), "µJ*s")
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-201")
                for i in range(len(optimal_x)):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDeFiNES Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tErr ", optimal_obj[i,0].item(), "%")
                    print("\tEDP: ",(-1*optimal_obj[i,1].item()), "µJ*s")
        
        def _to_2d_numpy(x):
            # 统一成 numpy 2D
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x, dtype=object)  # 允许混合类型（必要时再转）
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return x

        def _infer_fmt(arr: np.ndarray):
            # 单一 dtype 的数组走统一 fmt；混合列时回退到按列推断
            if arr.dtype.kind in ("i", "u"):     # 整型
                return "%d"
            if arr.dtype.kind == "f":            # 浮点
                return "%.6f"                    # 自行调整小数位
            if arr.dtype.kind in ("U", "S", "O"):# 字符串/混合
                return "%s"
            return "%s"

        def save_csv_no_header(path: str, data):
            arr = _to_2d_numpy(data)
            # 如果是混合类型（object），尝试逐列降为具体类型
            if arr.dtype == object:
                cols = []
                for j in range(arr.shape[1]):
                    col = np.asarray(arr[:, j])
                    # 尝试转成 float/int
                    try:
                        if np.all([str(v).isdigit() for v in col]):
                            col = col.astype(np.int64)
                        else:
                            col = col.astype(np.float64)
                    except Exception:
                        col = col.astype(str)
                    cols.append(col)
                arr = np.column_stack(cols)

            fmt = _infer_fmt(arr)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savetxt(path, arr, delimiter=",", fmt=fmt, header="", comments="")

        base_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' + self.set_acqu_algo + '_MODEL_ID_' + self.model_id.split("/")[1] + '_SEED_' + str(self.seed) + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_")
        os.makedirs(base_path, exist_ok=True)

        logs_path = os.path.join(base_path, "Logs")
        os.makedirs(logs_path, exist_ok=True)
        
        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/hvs.csv'
        # np.savetxt(save_path, hvs)
        # save_csv_no_header(base_path + "/legacy_hvs.csv", hvs)
        
        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/train_output.csv'
        # np.savetxt(save_path, train_obj.cpu().numpy())
        save_csv_no_header(base_path + "/[Legacy]train_output.csv", train_obj)
        
        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/train_input.csv'
        # np.savetxt(save_path, train_x.cpu().numpy())
        save_csv_no_header(base_path + "/[Legacy]train_input.csv", train_x)

        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/opt_efficiency_analys.csv'
        # np.savetxt(save_path, self.search_time_rec.cpu().numpy())
        save_csv_no_header(base_path + "/time_used_list.csv", torch.tensor(self.time_used_list, **tkwargs))
        
        # if all(v is not None for v in (self.archiv_train_x, self.archiv_train_obj)):
        #     # save_csv_no_header(base_path + "/ture_pareto_arch.csv", torch.tensor(self.archiv_train_x, **tkwargs))
        #     # save_csv_no_header(base_path + "/ture_pareto_front.csv", torch.tensor(self.archiv_train_obj, **tkwargs))
        #     save_csv_no_header(base_path + "/ture_pareto_arch.csv", self.archiv_train_x.detach().cpu())
        #     save_csv_no_header(base_path + "/ture_pareto_front.csv", self.archiv_train_obj.detach().cpu())

        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/opt_efficiency_analys.csv'
        # np.savetxt(save_path, self.search_time_rec.cpu().numpy())
        # save_csv_no_header(base_path + "/opt_efficiency_analys.csv", self.search_time_rec)

        # visit_time = torch.tensor([[self.estm_vist_time, self.TOTAL_RUN_TIME]], **tkwargs)
        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/comput_cost_analys.csv'
        # np.savetxt(save_path, visit_time.cpu().numpy())
        # save_csv_no_header(base_path + "/comput_cost_analys.csv", visit_time.cpu().numpy())
        
        # save_path = 'COFleX_result/' + self.Hardware_Arch + '_LORA_' + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/opt_vs_time_analys.csv'
        # np.savetxt(save_path, self.OPT_VS_TIME_REC.cpu().numpy())
        # save_csv_no_header(base_path + "/opt_vs_time_analys.csv", self.OPT_VS_TIME_REC)

        # assert self.TOTAL_TRAIN_OBJ_REC.size(0) == self.TOTAL_TRAIN_X_REC.size(0), "train_x and train_obj should have the same number of rows"
        # out = torch.cat(
        #     [self.TOTAL_TRAIN_X_REC.to(self.TOTAL_TRAIN_OBJ_REC.device, dtype=self.TOTAL_TRAIN_OBJ_REC.dtype), self.TOTAL_TRAIN_OBJ_REC],
        #     dim=1
        # )
        # save_csv_no_header(base_path + "/all_train_x/obj_rec.csv", out.cpu().numpy())
        
        self.plots(hvs, train_obj)

        self.cleanup_proxy_ckpt()

        return train_obj, train_x

    def pareto_check(self, iteration, X, y):
        P = []  
        F = defaultdict(list)  
        for i in range(X.shape[0]):
            P.append(Individual())
            P[i].solution = X[i] 
            P[i].calculate_objective(y[i])  
        F = self.fast_non_dominated_sort(P)
        for i in range(1, len(F) + 1):
            if not F[i]:  # Check if L is an empty list
                # print("_SKIP_")
                continue
            self.crowding_distance_assignment(F[i])  
            F[i].sort(key=lambda x: x.distance)  
        P = []
        for i in range(min(self.cluster_lim, len(F))):
            if F[i]:
                t = []
                t.append(F[i])
                P.extend(t)
                # plt.clf()
        plt.title('current generation:' + str(iteration))
        self.plot_P(P)
        for t in (P):  
            p_err = [ind.objective[0] for ind in t]
            p_eng = [ind.objective[1] for ind in t]
            p_x = [ind.solution for ind in t]
        return p_err, p_eng, p_x
    
    # -------- Pareto 前沿（2D）索引：默认两维最小化 --------
    def pareto_front_2d_indices(self, X, Y, maximize_x=False, maximize_y=False):
        x = -np.asarray(X, dtype=np.float64).reshape(-1) if maximize_x else np.asarray(X, dtype=np.float64).reshape(-1)
        y = -np.asarray(Y, dtype=np.float64).reshape(-1) if maximize_y else np.asarray(Y, dtype=np.float64).reshape(-1)

        assert x.shape == y.shape, "X/Y 长度不一致"
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        keep_indices = np.nonzero(mask)[0]

        # 扫描线：按 x 升序，维持当前最优 y
        order = np.argsort(x, kind="mergesort")
        best = np.inf
        pf_local = []
        for i in order:
            if y[i] < best:
                pf_local.append(i)
                best = y[i]
        return keep_indices[np.array(pf_local, dtype=int)]

    # -------- 读数据并绘图 --------
    def plot_pareto_acqu_algo(
        self,
        iteration,
        X,
        y,
        
        old_X=None,
        old_y=None,

        new_add_X=None,
        new_add_y=None,

        maximize_x=False,
        maximize_y=False,
        out_path="Figs/pf_coflex_.png",
        id=0,
        ):

        new_pairs = None

        title=f"Pareto Front for {self.acqu_algo}",

        safe_model_id = self.model_id.replace("/", "_")

        # 全局统一，局外不同
        if self.ts is None:
            self.ts = datetime.now().strftime("_%m_%d_%H_%M_")

        out_path=f"Figs/{self.acqu_algo}_{safe_model_id}_{self.Hardware_Arch}_{self.seed}_{self.ts}/pf_iter={iteration}_id={id}.png"
        
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
        x_col_name = "PPL"
        y_col_name = 'Latency(s)'

        Xs = X
        Y_err = y[:, 0]
        Y_eng = y[:, 1]

        pf_idx = self.pareto_front_2d_indices(Y_err, Y_eng, maximize_x, maximize_y)

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        # 全体点（浅灰、无描边）
        ax.scatter(Y_err, Y_eng, c="#BDBDBD", s=28, alpha=0.75, linewidths=0, label="All Points")

        if new_add_y is not None:
            curr_pairs = torch.as_tensor(new_add_y[:, :2], **tkwargs)

            if self.new_add_Y_err is None:      # 第一次
                self.new_add_Y_err = curr_pairs[:, 0].clone()
                self.new_add_Y_eng = curr_pairs[:, 1].clone()
                new_pairs = curr_pairs          # 第一轮所有点都算“新点”
            else:
                prev_pairs = torch.stack([self.new_add_Y_err, self.new_add_Y_eng], dim=1)  # (N, 2)
                eq = (curr_pairs.unsqueeze(1) == prev_pairs.unsqueeze(0)).all(dim=2)       # (K, N)
                exists = eq.any(dim=1)
                mask = ~exists
                new_pairs = curr_pairs[mask]

                if new_pairs.numel() > 0:
                    self.new_add_Y_err = torch.cat([self.new_add_Y_err, new_pairs[:, 0]], dim=0)
                    self.new_add_Y_eng = torch.cat([self.new_add_Y_eng, new_pairs[:, 1]], dim=0)

            if self.new_add_Y_err is not None:
                ax.scatter(
                    self.new_add_Y_err.detach().cpu().numpy(),
                    self.new_add_Y_eng.detach().cpu().numpy(),
                    c="#F88181",
                    s=12,
                    alpha=0.4,
                    linewidths=0,
                    label="All New Points (history)",
                )

        # 只要这一轮真有“新东西”，就画红点
        if (new_pairs is not None) and (new_pairs.numel() > 0):
            ax.scatter(
                new_pairs[:, 0].detach().cpu().numpy(),
                new_pairs[:, 1].detach().cpu().numpy(),
                c="#FF0000",
                s=30,
                alpha=0.75,
                linewidths=0,
                label="New Added Points",
            )

        pf = np.column_stack([Y_err[pf_idx], Y_eng[pf_idx]])
        pf = pf[np.argsort(pf[:, 0])]
        ax.scatter(pf[:,0], pf[:,1],
            c="#444444", s=38, zorder=3,
            edgecolors="white", linewidths=0.6,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="white")],
            label="Pareto Front")
        ax.plot(pf[:,0], pf[:,1], linestyle="-.", color="dimgray", lw=2)

        xlab = f"{x_col_name} ↓" if not maximize_x else f"{x_col_name} ↑"
        ylab = f"{y_col_name} ↓" if not maximize_y else f"{y_col_name} ↑"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0., frameon=True, framealpha=0.9)

        plt.tight_layout()
        fig.savefig(out_path, dpi=900)

        print(f"[✔] All Pareto Fronts Figs saved to: {out_path}")

        plt.close(fig)

        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if old_X is not None:
        #     X_array = np.asarray(old_X)
        #     assert X_array.shape[1] == 10, f"Expected X to have 10 columns, got {X_array.shape[1]}"

        #     x_group_names = ["pos", "rank", "prec_sw", "layer"]
        #     y_group_names = ["prec_hw", "nmp_c", "i_buff", "o_buff", "w_buf", "bw"]

        #     x_space_out_path = f"figs/{self.acqu_algo}/xspace_iter={iteration}_id={id}.png"
        #     os.makedirs(os.path.dirname(x_space_out_path) or ".", exist_ok=True)

        #     fig, axes = plt.subplots(
        #         len(x_group_names),
        #         len(y_group_names),
        #         figsize=(18, 12),
        #         sharex="col",
        #         sharey="row",
        #     )

        #     new_add_X_array = None
        #     if new_add_X is not None:
        #         new_add_X_array = np.asarray(new_add_X)

        #     for i, x_name in enumerate(x_group_names):
        #         for j, y_name in enumerate(y_group_names):
        #             ax = axes[i, j]
        #             ax.scatter(
        #                 X_array[:, 4 + j],
        #                 X_array[:, i],
        #                 c="#2600FFFF",
        #                 s=16,
        #                 alpha=0.7,
        #                 linewidths=0,
        #             )
        #             if new_add_X_array is not None:
        #                 ax.scatter(
        #                     new_add_X_array[:, 4 + j],
        #                     new_add_X_array[:, i],
        #                     c="#FF0000",
        #                     s=22,
        #                     alpha=0.85,
        #                     linewidths=0,
        #                 )
        #             if i == len(x_group_names) - 1:
        #                 ax.set_xlabel(y_name)
        #             if j == 0:
        #                 ax.set_ylabel(x_name)

        #     plt.tight_layout()
        #     fig.savefig(x_space_out_path, dpi=900)
        #     print(f"[✔] X Space Figs saved to: {x_space_out_path}")
        #     plt.close(fig)

        return Y_err[pf_idx], Y_eng[pf_idx], Xs[pf_idx]
    
    def acquisition_function(self, gp, x, y_min, scaler=None, levels=None):
        if levels is not None:
            x = np.asarray(x, dtype=float)
            snapped = np.empty_like(x)
            for j, lv in enumerate(levels):
                lv_arr = np.asarray(lv, dtype=float)
                dist = np.abs(x[:, [j]] - lv_arr[None, :])
                snapped[:, j] = lv_arr[np.argmin(dist, axis=1)]
            x = snapped
        if scaler is not None:
            x = scaler.transform(x)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        z = (y_min - mu) / sigma
        ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    

    def optimize_acquisition(self, gp, X, bounds, y_min, scaler=None, levels=None): # the best performance y and random chose x
        def objective(X):
            X = X.reshape(1, -1)
            return -self.acquisition_function(gp, X, y_min, scaler=scaler, levels=levels)

        n_starts = min(8, len(X))
        start_indices = np.random.choice(len(X), size=n_starts, replace=False)

        best_x = None
        best_val = float("inf")
        for idx in start_indices:
            result = minimize(
                objective,
                x0=X[idx],
                bounds=bounds,
                method="L-BFGS-B",
            )
            val = float(result.fun) if np.isfinite(result.fun) else float("inf")
            if val < best_val:
                best_val = val
                best_x = result.x

        if best_x is None:
            best_x = X[random.randint(0, len(X)-1)]
        return best_x


class Individual(object):
    def __init__(self):
        self.solution = None  
        self.objective = defaultdict()
        self.n = 0  
        self.rank = 0  
        self.S = []  
        self.distance = 0  
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun
    def __lt__(self, other):
        v1 = self.objective  
        v2 = other.objective  
        for i in range(len(v1)):
            if v1[i] > v2[i]:  
                return 0  
        return 1