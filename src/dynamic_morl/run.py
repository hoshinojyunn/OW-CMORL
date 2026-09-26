import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals', 'baselines'))
sys.path.append(os.path.join(base_dir, 'externals', 'pytorch-a2c-ppo-acktr-gail'))

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl import morl
import torch
import gym
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Logger:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.logfile.write(data)
        self.logfile.flush()

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            if self.logfile is not None and not self.logfile.closed:
                self.logfile.flush()
        except Exception:
            pass

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)

def main():
    torch.set_default_dtype(torch.float64)
    if hasattr(torch, "multiprocessing") and hasattr(torch.multiprocessing, "set_sharing_strategy"):
        try:
            # `file_system` spawns `torch_shm_manager` helpers, which were leaving
            # high-CPU orphan processes behind on WSL after interrupted runs.
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
        except RuntimeError:
            pass
    
    # ppo parameters
    args_list = ['--lr', '3e-4',
                 '--selection-method', 'online-window',
                 '--strict-online-context',
                 '--use-linear-lr-decay',
                 '--use-gae',
                 '--gae-lambda', '0.95',
                 '--entropy-coef', '0',
                 '--value-loss-coef', '0.5',
                 '--num-steps', '512', #512
                 '--num-processes', '8',
                 '--ppo-epoch', '10',
                 '--num-mini-batch', '32',
                 '--use-proper-time-limits',
                 '--ob-rms',
                 '--obj-rms',
                 '--raw']

    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    # build saving folder
    save_dir = args.save_dir
    try:
        os.makedirs(save_dir, exist_ok = True)
    except OSError:
        pass
    
    # output arguments
    fp = open(os.path.join(save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    logfile = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    old_stdout = sys.stdout
    sys.stdout = Logger(sys.stdout, logfile)
    try:
        morl.run(args)
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        logfile.close()

if __name__ == "__main__":
    main()
