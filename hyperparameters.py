import argparse
from models.models import MODEL_FACTORIES

config1 = {
    'optimizer': 'adam',
    'data': 'mnist',
    'lr': 0.001,
    'model': 'cnn',
    'max_steps': 1000,
    'n_samples': 0,
    'cross_val': 5,
    'batch_size': 64,
    "n_vocab": 20000,
    "max_len": 100
}


def parse_cmd_args():
    """
    This function sets the command line arguments to look for. The defaults are given in config1 above.
    :return:
    """
    parser = argparse.ArgumentParser(description='This program applies an RL method to an OpenAI gym environment')
    for name, val in config1.items():
        if type(val) is bool:
            parser.add_argument('--' + name, action='store_true', dest=name)
            parser.add_argument('--not_' + name, action='store_false', dest=name)
            parser.set_defaults(**{name: val})
        else:
            parser.add_argument('--' + name, type=type(val), default=val)

    args = parser.parse_args()
    return args


class HyperParameters():

    def __init__(self, args):
        if isinstance(args, dict):
            self.__dict__ = args
            return
        self.optimizer = args.optimizer
        self.data = args.data
        self.lr = args.lr
        self.model = MODEL_FACTORIES[args.model]
        self.max_steps = args.max_steps
        self.n_samples = args.n_samples
        self.cross_val = args.cross_val
        self.batch_size = args.batch_size
        self.n_vocab = args.n_vocab
        self.max_len = args.max_len