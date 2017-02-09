import argparse
from models.models import MODEL_FACTORIES

config1 = {
    'optimizer': 'adam',
    'data': 'mnist',
    'lr': 0.001,
    'model': 'cnn',
    'max_steps': 1000
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