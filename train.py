from experiment_util import init_log_dir
from hyperparameters import HyperParameters, parse_cmd_args

if __name__ == "__main__":
    hp = HyperParameters(parse_cmd_args())
    logdir = init_log_dir(hp)