import os
import logging
import wandb
import sys
import submitit
import torch
from torch import distributed as dist
from dataclasses import dataclass, field, is_dataclass
import pathlib
from submitit import SlurmExecutor
from simple_parsing import ArgumentParser, subgroups, Serializable
import typing as tp


# dataclass built in asdict doesn't work with submitit's serialization
# as a hack, we use this function instead
def asdict(dataclass):
    out = {}
    for k, v in dataclass.__dict__.items(): 
        if is_dataclass(v): 
            out[k] = asdict(v)
        else:
            out[k] = v
    return out


def slurm_checkpoint_dir():
    """
    Returns the path to the slurm checkpoint directory if running on a slurm cluster,
    otherwise returns None. (This function is designed to work on the vector cluster)
    """
    import os

    if "SLURM_JOB_ID" not in os.environ:
        return None
    return os.path.join("/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"])


def generate_experiment_name():
    """
    Generates a fun experiment name.
    """
    from coolname import generate_slug
    from datetime import datetime

    return f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-{generate_slug(2)}'


def generate_exp_dir(name=None, project=None, group=None):
    """
    Generates a directory name for the experiment.
    """
    exp_name = name or generate_experiment_name()
    exp_dir = "logs"
    # if project is not None:
    #     exp_dir = os.path.join(exp_dir, project)
    if group is not None:
        exp_dir = os.path.join(exp_dir, group)
    exp_dir = os.path.join(exp_dir, exp_name)
    return exp_dir

@dataclass
class SubmititJobSubmissionConfig:
    """Configuration for running the job in a slurm cluster using submitit."""

    timeout_min: int = 60 * 4
    slurm_gres: str = "gpu:a40:1"
    mem_gb: int = 16
    cpus_per_task: int = 16
    slurm_qos: tp.Literal["normal", "m2", "m3", "m4", "deadline"] = "m2"
    slurm_account: tp.Optional[str] = None
    slurm_exclude: tp.Optional[str] = None
    slurm_setup: list[str] = field(default_factory=lambda: [
        "module load pytorch2.1-cuda11.8-python3.10",
    ])

# might use this later
@dataclass
class SlurmJobConfig: 
    gres: str = "gpu:a40:1"
    mem: str = "16GB"
    cpus_per_task: int = 16
    gpus_per_task: int = 1
    qos: tp.Literal["normal", "m2", "m3", "m4"] = "m2"
    time: str = "4:00:00"
    ntasks_per_node: int = 1
    nodes: int = 1
    setup: list[str] = field(default_factory=lambda: [
        "module load pytorch2.1-cuda11.8-python3.10",
    ])


@dataclass
class LocalJobSubmissionConfig:
    """Configuration for running the job locally"""


@dataclass
class BasicExperimentConfig(Serializable):
    """
    Basic configuration for the experiment.
    """

    exp_dir: str = None
    name: str = None
    group: str = None
    project: str = None
    entity: str = "mahdigilany"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    cluster: SubmititJobSubmissionConfig | LocalJobSubmissionConfig = subgroups(
        {"slurm": SubmititJobSubmissionConfig, "local": LocalJobSubmissionConfig},
        default="local",
    )

    def __post_init__(self):
        if self.name is None:
            self.name = generate_experiment_name()
        if self.exp_dir is None:
            self.exp_dir = generate_exp_dir(self.name, self.project, self.group)


class BasicExperiment:
    """
    Base class for an experiment. Handles boilerplate setup such
    as logging, experiment and checkpoint directory setup, and
    enables automatic argument parsing and submission to a cluster.

    Example usage:
    ```
    @dataclass
    class MyExperimentConfig(BasicExperimentConfig):
        # add your config options here
        pass

    class MyExperiment(BasicExperiment):
        config_class = MyExperimentConfig
        config: MyExperimentConfig

        def setup(self):
            super().setup()
            # add your setup code here

        def __call__(self):
            # implement your experiment here
            pass

    if __name__ == '__main__':
        MyExperiment.submit() # handles argument parsing and submission to cluster
    """

    config_class = BasicExperimentConfig
    config: BasicExperimentConfig

    def __init__(self, config: config_class):
        self.config = config

    def __call__(self):
        """
        Runs the experiment.
        """
        raise NotImplementedError

    def setup(self):
        os.makedirs(self.config.exp_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.config.exp_dir, "config.yaml")):
            with open(os.path.join(self.config.exp_dir, "config.yaml"), "w") as f:
                import yaml
                yaml.dump(asdict(self.config), f)
        
        if not os.path.exists(os.path.join(self.config.exp_dir, "checkpoints")):
            if slurm_checkpoint_dir() is not None:
                # sym link slurm checkpoints dir to local checkpoints dir
                os.symlink(
                    slurm_checkpoint_dir(),
                    os.path.join(self.config.exp_dir, "checkpoints"),
                )
            else:
                os.makedirs(os.path.join(self.config.exp_dir, "checkpoints"))
        ckpt_dir = os.path.join(self.config.exp_dir, "checkpoints")

        stdout_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(os.path.join(self.config.exp_dir, "out.log"))
        logging.basicConfig(
            level=logging.INFO, handlers=[stdout_handler, file_handler], force=True
        )

        # also log tracebacks with excepthook
        def excepthook(type, value, tb):
            logging.error("Uncaught exception: {0}".format(str(value)))
            import traceback

            traceback.print_tb(
                tb, file=open(os.path.join(self.config.exp_dir, "out.log"), "a")
            )
            logging.error(f"Exception type: {type}")
            sys.__excepthook__(type, value, tb)

        sys.excepthook = excepthook

        if self.config.resume and "wandb_id" in os.listdir(self.config.exp_dir):
            wandb_id = (
                open(os.path.join(self.config.exp_dir, "wandb_id")).read().strip()
            )
            logging.info(f"Resuming wandb run {wandb_id}")
        else:
            wandb_id = wandb.util.generate_id()
            open(os.path.join(self.config.exp_dir, "wandb_id"), "w").write(wandb_id)

        if not self.config.use_wandb:
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            entity=self.config.entity,
            project=self.config.project
            if not self.config.debug  
            else f"{self.config.project}-debug",
            group=self.config.group,
            config=asdict(self.config),
            resume="allow",
            name=os.path.basename(self.config.exp_dir),
            id=wandb_id,
            dir=ckpt_dir,
        )
        self.ckpt_dir = ckpt_dir

    def checkpoint(self):
        """
        Handles checkpointing the experiment when running on a cluster
        using submitit. This method is called by submitit when the job is preempted or times out.
        """
        logging.info(f"Handling Preemption or timeout!")
        from submitit.helpers import DelayedSubmission

        new_job = self.__class__(self.config)
        logging.info(f"Resubmitting myself.")
        return DelayedSubmission(new_job)

    @staticmethod
    def get_submitit_executor(config): 
        if isinstance(config.cluster, LocalJobSubmissionConfig): 
            return None 
        
        elif isinstance(config.cluster, SlurmJobConfig):
            executor = SlurmExecutor(
                folder=os.path.join(config.exp_dir, "submitit_logs"),
                max_num_timeout=10,
            )
            executor.update_parameters(**asdict(config.cluster))
            return executor

        elif isinstance(config.cluster, SubmititJobSubmissionConfig):
            executor = submitit.AutoExecutor(
                folder=os.path.join(config.exp_dir, "submitit_logs"),
                slurm_max_num_timeout=10,
            )

            executor.update_parameters(**asdict(config.cluster))
            return executor
        else:
            raise ValueError(f"Invalid cluster type: {config.cluster}")

    @classmethod 
    def parse_args(cls): 
        parser = ArgumentParser()
        parser.add_arguments(cls.config_class, dest="config")
        return parser.parse_args().config

    @classmethod
    def submit(cls, cfg=None):
        cfg = cls.parse_args() if cfg is None else cfg
        executor = cls.get_submitit_executor(cfg)
        job = cls(cfg)
       
        if executor is not None: 
            job = executor.submit(job)
            print(f"Submitted job: {job.job_id}")
            print(f"Outputs: {job.paths.stdout}")
            print(f"Errors: {job.paths.stderr}")
        else: 
            job()
