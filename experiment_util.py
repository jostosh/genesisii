import os


def init_log_dir(hp, params_to_reflect=['optimizer', 'data'], base=None):
    if not base:
        base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
    log_sub_root = os.path.join(base, *['{}={}'.format(param, hp.__dict__[param]) for param in params_to_reflect])

    os.makedirs(log_sub_root, exist_ok=True)
    dirs = os.listdir(log_sub_root)

    logdir = os.path.join(log_sub_root, 'run%03d' % (int(sorted(dirs)[-1][-3:]) + 1,) if dirs else 'run000')
    os.makedirs(logdir)

    return logdir