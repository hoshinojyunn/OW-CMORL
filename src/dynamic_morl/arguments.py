import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='RL')

    # MORL parameters
    parser.add_argument('--env-name',
        default='building_3d_dynamic',
        help='environment to train on')
    parser.add_argument('--obj-num',
        type=int,
        default=2,
        help='number of objectives')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=5e6,
        help='number of environment steps to train (default: 5e6)')
    parser.add_argument('--num-tasks',
        type=int,
        default=6,
        help='number of rl task in each epoch')
    parser.add_argument('--seed', 
        type=int, 
        default=0, 
        help='random seed (default: 1)')
    parser.add_argument('--min-weight',
        type=float,
        default=0.0,
        help='minimum of weight range')
    parser.add_argument('--max-weight',
        type=float,
        default=1.0,
        help='maximum of weight range')
    parser.add_argument('--delta-weight',
        type=float,
        default=0.2,
        help='granularity of weight combinations in warm-up stage')
    parser.add_argument('--eval-delta-weight',
        type=float,
        default=0.01,
        help='granularity of weight combinations in evaluation')
    parser.add_argument('--warmup-iter',
        type=int,
        default=80,
        help='number of RL iterations to run for warm up')
    parser.add_argument('--update-iter',
        type=int, 
        default=20,
        help='number of RL iterations between evolutionary processes')
    parser.add_argument('--eval-num',
        type=int,
        default=10,
        help='number of fitness evaluation times')
    parser.add_argument('--selection-method',
        type=str,
        default='online-window',
        help='which type of method to select the tasks in the population [online-window, dynamic-frontier, crowding, random]')
    parser.add_argument('--pbuffer-num',
        type=int,
        default=200,
        help='number of performance buffers')
    parser.add_argument('--pbuffer-size',
        type=int,
        default=2,
        help='size of each performance buffer')
    parser.add_argument('--num-weight-candidates',
        type=int,
        default=7,
        help='number of weight candiates for each population sample')
    parser.add_argument('--sparsity',
        default=1.0,
        type=float,
        help='alpha of sparsity metrics')
    parser.add_argument('--obj-rms',
        default=False,
        action='store_true',
        help='if use running mean std on objectives')
    parser.add_argument('--ob-rms',
        default=False,
        action='store_true',
        help='if use running mean std on observations')
    parser.add_argument('--raw',
        default=False,
        action='store_true',
        help='use undiscounted evaluation')
    parser.add_argument(
        '--rl-log-interval',
        type=int,
        default=10,
        help='RL log interval, one log per n updates')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--update-method',
        type=str,
        default='cmorl-ipo',
        help='')
    parser.add_argument(
        '--num-select',
        type=int,
        default=5,
        help='')
    parser.add_argument(
        '--ref-point', 
        type=float, 
        nargs='+', 
        default=[0., 0.])
    parser.add_argument('--auto-ref-point',
        action='store_true',
        default=False,
        help='calibrate Pareto reference point from warm-up policies')
    parser.add_argument(
        '--num-time-steps',
        type=int,
        default=2500000,
        help='')
    parser.add_argument(
        '--num-init-steps',
        type=int,
        default=1500000,
        help='')
    parser.add_argument('--cost-objective',
        default=False,
        action='store_true',
        help='')
    parser.add_argument('--drift-window',
        type=int,
        default=6,
        help='length of environment-context window for drift attention')
    parser.add_argument('--drift-steps',
        type=int,
        default=4,
        help='gradient steps for the context encoder per update')
    parser.add_argument('--strict-online-context',
        action='store_true',
        default=False,
        help='disable oracle regime ids inside the adaptation loop and infer shifts only from history windows')
    parser.add_argument('--context-history-size',
        type=int,
        default=10,
        help='number of past context snapshots retained by the online context memory')
    parser.add_argument('--context-probe-samples',
        type=int,
        default=3,
        help='number of deployed expert traces used to build the current online context window')
    parser.add_argument('--context-trace-steps',
        type=int,
        default=6,
        help='number of recent environment-context tokens taken from each deployed expert trace when building the online context window')
    parser.add_argument('--context-sample-mode',
        type=str,
        default='salient_mix',
        choices=['tail', 'uniform', 'salient_mix'],
        help='how to sample tokens from the recent online trace when building the context window')
    parser.add_argument('--context-saliency-alpha',
        type=float,
        default=0.35,
        help='weight of context-change saliency when context-sample-mode=salient_mix')
    parser.add_argument('--context-buffer-steps',
        type=int,
        default=48,
        help='maximum number of aggregated online context steps retained across iterations')
    parser.add_argument('--context-forecast-lambda',
        type=float,
        default=0.50,
        help='weight of forecasted future context when building the online target context')
    parser.add_argument('--context-history-lambda',
        type=float,
        default=0.10,
        help='weight of historical context memory when smoothing the online target context')
    parser.add_argument('--context-nearest-k',
        type=int,
        default=3,
        help='number of nearest historical context snapshots used to smooth the online objective-gap estimate')
    parser.add_argument('--bank-num-slots',
        type=int,
        default=6,
        help='maximum number of online context expert slots kept in the expert bank')
    parser.add_argument('--bank-slot-size',
        type=int,
        default=3,
        help='maximum number of experts retained per context slot in the expert bank')
    parser.add_argument('--bank-merge-threshold',
        type=float,
        default=0.82,
        help='cosine-affinity threshold for merging a new expert into an existing context slot')
    parser.add_argument('--bank-query-slots',
        type=int,
        default=3,
        help='number of nearest context slots queried during online routing')
    parser.add_argument('--bank-query-topk',
        type=int,
        default=2,
        help='number of experts retrieved per queried slot during online routing')
    parser.add_argument('--bank-enable',
        type=int,
        default=1,
        help='whether to enable the online context expert bank (1=yes, 0=no)')
    parser.add_argument('--bank-retrieval-backend',
        choices=['hnsw', 'exact'],
        default='hnsw',
        help='context-slot retrieval backend; exact is the brute-force HNSW ablation')
    parser.add_argument('--hnsw-ef-search',
        type=int,
        default=32,
        help='HNSW search breadth used by the context expert bank')
    parser.add_argument('--hnsw-m',
        type=int,
        default=16,
        help='HNSW graph degree used by the context expert bank')
    parser.add_argument('--hnsw-ef-construction',
        type=int,
        default=100,
        help='HNSW construction breadth used by the context expert bank')
    parser.add_argument('--bank-query-mode',
        choices=['context', 'global-best'],
        default='context',
        help='context routes by the encoded query; global-best ignores context for the ablation')
    parser.add_argument('--context-encoder-enable',
        type=int,
        default=1,
        help='whether to train/use the context Transformer encoder (0 for no-context ablation)')
    parser.add_argument('--online-shift-threshold',
        type=float,
        default=0.6,
        help='threshold for unsupervised shift detection from utility/objective changes')
    parser.add_argument('--online-shift-min-gap',
        type=int,
        default=6,
        help='minimum temporal spacing between two detected online shifts')
    parser.add_argument('--chlor-episode-length',
        type=int,
        default=288,
        help='episode length for ChlorAlkaliEnv')
    parser.add_argument('--chlor-regime-clusters',
        type=int,
        default=20,
        help='number of representative continuous-context regime clusters used for ChlorAlkali evaluation')
    parser.add_argument('--fine-regime-clusters',
        type=int,
        default=20,
        help='number of fine-grained regime clusters used by the shared benchmark protocol')
    parser.add_argument('--fine-regime-catalog-episodes',
        type=int,
        default=256,
        help='number of probe episodes used to build the fine-regime catalog')
    parser.add_argument('--fine-regime-context-steps',
        type=int,
        default=4,
        help='number of initial context steps averaged when assigning an episode to a fine regime')
    parser.add_argument('--dynamic-lambda',
        type=float,
        default=0.75,
        help='weight of drift-aware regime affinity during archive selection')
    parser.add_argument('--knee-lambda',
        type=float,
        default=0.3,
        help='weight of knee score during archive selection')
    parser.add_argument('--resilience-lambda',
        type=float,
        default=1.75,
        help='weight of shift-recovery resilience during archive selection')
    parser.add_argument('--resilience-recovery-weight',
        type=float,
        default=0.30,
        help='recovery-score contribution inside the resilience term')
    parser.add_argument('--resilience-regret-weight',
        type=float,
        default=0.20,
        help='shift-regret contribution inside the resilience term')
    parser.add_argument('--resilience-latency-weight',
        type=float,
        default=0.15,
        help='recovery-latency contribution inside the resilience term')
    parser.add_argument('--resilience-utility-weight',
        type=float,
        default=0.10,
        help='utility contribution inside the resilience term')
    parser.add_argument('--resilience-gap-weight',
        type=float,
        default=0.10,
        help='pre/post utility-gap contribution inside the resilience term')
    parser.add_argument('--resilience-bank-weight',
        type=float,
        default=0.15,
        help='expert-bank score contribution inside the resilience term')
    parser.add_argument('--diversity-lambda',
        type=float,
        default=0.10,
        help='weight of archive diversity during dynamic selection')
    parser.add_argument('--dynamic-affinity-weight',
        type=float,
        default=0.65,
        help='context-affinity contribution inside the dynamic routing signal')
    parser.add_argument('--dynamic-freshness-weight',
        type=float,
        default=0.20,
        help='recency contribution inside the dynamic routing signal')
    parser.add_argument('--dynamic-bank-weight',
        type=float,
        default=0.15,
        help='expert-bank contribution inside the dynamic routing signal')
    parser.add_argument('--shift-gap-lambda',
        type=float,
        default=1.0,
        help='weight of objective-level shift gaps during extension scheduling')
    parser.add_argument('--repeat-topk',
        type=int,
        default=2,
        help='number of top-priority objectives to repeat during dynamic extension')
    parser.add_argument('--final-archive-mode',
        type=str,
        default='pareto',
        choices=['pareto', 'union', 'resilient-diverse'],
        help='how to export the final candidate archive before shared-protocol evaluation')
    parser.add_argument('--final-archive-max-samples',
        type=int,
        default=64,
        help='maximum number of deduplicated samples kept when final-archive-mode=union')
    parser.add_argument('--expert-bank-recovery-weight',
        type=float,
        default=1.0,
        help='recovery contribution inside the expert-bank sample score')
    parser.add_argument('--expert-bank-regret-weight',
        type=float,
        default=0.45,
        help='shift-regret contribution inside the expert-bank sample score')
    parser.add_argument('--expert-bank-latency-weight',
        type=float,
        default=0.25,
        help='recovery-latency contribution inside the expert-bank sample score')
    parser.add_argument('--expert-bank-gap-weight',
        type=float,
        default=0.10,
        help='pre/post utility-gap contribution inside the expert-bank sample score')
    parser.add_argument('--final-score-recovery-weight',
        type=float,
        default=1.2,
        help='recovery contribution inside the final union-archive score')
    parser.add_argument('--final-score-regret-weight',
        type=float,
        default=0.35,
        help='shift-regret contribution inside the final union-archive score')
    parser.add_argument('--final-score-latency-weight',
        type=float,
        default=0.25,
        help='recovery-latency contribution inside the final union-archive score')
    parser.add_argument('--final-score-bank-weight',
        type=float,
        default=0.15,
        help='expert-bank score contribution inside the final union-archive score')
    parser.add_argument('--final-score-affinity-weight',
        type=float,
        default=0.10,
        help='bank-query affinity contribution inside the final union-archive score')
    parser.add_argument('--final-score-train-iter-weight',
        type=float,
        default=0.05,
        help='training-iteration recency contribution inside the final union-archive score')
    parser.add_argument('--final-objective-diversity-lambda',
        type=float,
        default=0.35,
        help='objective-coverage contribution inside resilient-diverse final export')
    parser.add_argument('--final-context-diversity-lambda',
        type=float,
        default=0.35,
        help='context-coverage contribution inside resilient-diverse final export')
    parser.add_argument('--regime-schedule',
        type=str,
        default='cyclic',
        help='dynamic building regime schedule [cyclic, random]')
    parser.add_argument('--episodes-per-regime',
        type=int,
        default=4,
        help='episodes per building regime before switching')
    parser.add_argument('--ev-site',
        type=str,
        default='caltech',
        help='EVCharging site')
    parser.add_argument('--ev-periods',
        nargs='+',
        default=['Summer 2019', 'Spring 2020', 'Summer 2021'],
        help='EVCharging trace periods used as dynamic regimes')
    parser.add_argument('--ev-moer-forecast-steps',
        type=int,
        default=36,
        help='number of MOER forecast steps for EVCharging')
    parser.add_argument('--ev-disable-projection',
        action='store_true',
        default=False,
        help='disable EVCharging action projection')
    parser.add_argument('--cogen-renewables',
        nargs='+',
        type=float,
        default=[0.0, 10.0, 20.0],
        help='renewables magnitudes used as cogen dynamic regimes')
    parser.add_argument('--cogen-forecast-horizon',
        type=int,
        default=3,
        help='forecast horizon for cogen wrapper')
    parser.add_argument('--cogen-forecast-noise-std',
        type=float,
        default=0.0,
        help='forecast noise standard deviation for cogen wrapper')
    parser.add_argument('--sustaingym-building-weathers',
        nargs='+',
        default=['Hot_Dry', 'Warm_Marine', 'Mixed_Marine'],
        help='weather regimes for official SustainGym BuildingEnv')
    parser.add_argument('--regime-eval-samples',
        type=int,
        default=3,
        help='number of selected policies to re-evaluate per regime for dynamic metrics')
    parser.add_argument('--trace-recovery-window',
        type=int,
        default=12,
        help='window used to compute post-shift recovery metrics from evaluation traces')
    parser.add_argument('--trace-eval-samples',
        type=int,
        default=2,
        help='number of selected policies to evaluate on the dynamic trace for shift metrics')
    parser.add_argument('--shared-regime-eval-episodes',
        type=int,
        default=0,
        help='shared episode count used by the final benchmark protocol; 0 uses an environment-specific default')
    parser.add_argument('--shared-regime-seed-offset',
        type=int,
        default=0,
        help='extra deterministic seed offset used by the shared benchmark protocol')
    parser.add_argument(
        '--skip-final-shared-benchmark',
        action='store_true',
        default=False,
        help='save final policies and a resumable OOD runtime checkpoint without the offline shared-final sweep')
    parser.add_argument('--shared-plan-json',
        type=str,
        default='',
        help='optional shared regime plan JSON/payload used to force evaluation onto an exact regime/seed schedule')
    parser.add_argument('--env-config-json',
        type=str,
        default='',
        help='optional JSON file containing env_kwargs overrides for training/evaluation')
    parser.add_argument('--beta',
        default=0.9,
        type=float,
        help='constraint relax coefficient')
    parser.add_argument('--t',
        default=20,
        type=float,
        help='log barrier coefficient')
    parser.add_argument('--policy-buffer',
        default=200,
        type=int,
        help='policy buffer size')
    parser.add_argument('--eval-gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards in evaluation(default: 0.99)')
    parser.add_argument(
        '--rl-eval-interval',
        type=int,
        default=10,
        help='RL evaluation interval, one evaluation per n updates')

    # PPO parameters
    parser.add_argument(
        '--algo', default='ppo')
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument('--lr-decay-ratio',
        type=float,
        default=1,
        help='ratio of lr decay from beginning to the end (e.g., 1.0 means lr finally decays to 0, 0.0 means lr stays constant)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.995,
        help='discount factor for rewards (default: 0.995)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2048,
        help='timesteps per epoch.')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=4,
        help='how many training CPU processes to use (default: 4)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)') 
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--layernorm', 
        action='store_true',
        default=False,
        help='if use layernorm')

    return parser
