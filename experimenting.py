import subprocess
import concurrent.futures
from dataclasses import dataclass
from simple_parsing import ArgumentParser, subgroups, Serializable

from utils.setup import generate_exp_dir
from kernel_deeptime_experiment import KernelDeepTimeExpConfig, KernelDeepTimeExp
from deeptime_experiment import DeepTimeConfig


HORIZONS = [96, 192, 336, 720]

# deeptime
# LOOKBACKS_M_MULT = {
#     "exchange_rate": [1, 5, 7, 3],
#     "ettm2": [7, 5, 3, 1],
# }
# kernel-deeptime
LOOKBACKS_M_MULT = {
    "exchange_rate": [7, 5, 3, 2], # last one is not yet correct
    "ettm2": [7, 5, 3, 1], # last one is not yet correct
}


# class SeqKernelDeepTimeExp(KernelDeepTimeExp):
#     def __call__(self):
#         super().__call__()
        


@dataclass
class ExperimentConfig(Serializable):
    sub_exp: str = ""
    num_seeds: int = 1
    lookback_mult: str = None
    start_seed: int = 0
    exp_config: KernelDeepTimeExpConfig | DeepTimeConfig = subgroups(
        {"kernel_deeptime": KernelDeepTimeExpConfig, "deeptime": DeepTimeConfig},
        default="kernel_deeptime",
    )

def parsing_args(): 
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="config")
    return parser.parse_args().config


def submit_experiment(args, horizon, lookback_mult):
    data_name = args.exp_config.data_config.dataset_name
    target = "M" if args.exp_config.data_config.target_series_index is None else "S"
    lookback = lookback_mult * horizon
    
    args.exp_config.horizon = horizon
    args.exp_config.data_config.horizon = horizon
    args.exp_config.model_config.horizon = horizon
    args.exp_config.data_config.lookback = lookback
    
    for seed in range(args.start_seed, args.start_seed+args.num_seeds):
        if isinstance(args.exp_config, KernelDeepTimeExpConfig):
            args.exp_config.seed = seed
            args.exp_config.group = f"kernel-deeptime_{data_name}_in{lookback}_out{horizon}_{target}_sub{args.sub_exp}"
            args.exp_config.name = f"{args.exp_config.group}_{seed}"
            args.exp_config.exp_dir = generate_exp_dir(args.exp_config.name, args.exp_config.project, args.exp_config.group)
            KernelDeepTimeExp.submit(args.exp_config)
        else:
            raise ValueError("Not implemented yet")

# if __name__ == '__main__':
#     args: ExperimentConfig = parsing_args()

#     if args.exp_config.data_config.target_series_index is None:
#         LOOKBACKS = LOOKBACKS_M_MULT[args.exp_config.data_config.dataset_name]
#     else:
#         raise ValueError("Not implemented yet")

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(submit_experiment, args, horizon, lookback_mult)
#                    for horizon, lookback_mult in zip(HORIZONS, LOOKBACKS)]

#         # Wait for all futures to complete
#         for future in concurrent.futures.as_completed(futures):
#             future.result()  # Can handle exceptions if needed


if __name__ == '__main__': 
    args: ExperimentConfig = parsing_args()
    
    if (args.lookback_mult is None) or (args.lookback_mult=="None"):
        if args.exp_config.data_config.target_series_index is None:
            LOOKBACKS = LOOKBACKS_M_MULT[args.exp_config.data_config.dataset_name]
        else:
            raise ValueError("Not implemented yet")
    else:
        LOOKBACKS = list(map(int, args.lookback_mult.split(",")))
    
    for horizon, lookback_mult in zip(HORIZONS, LOOKBACKS):
        submit_experiment(args, horizon, lookback_mult)