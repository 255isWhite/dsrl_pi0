import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args
from general_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--task_id', default=-1, help='task id for libero environment', type=int)
    parser.add_argument('--task_suite', default='lift', help='task suite for libero/robomimic environment', type=str)
    parser.add_argument('--pi0_model', default='pi0_libero', help='which pi0 model to use', type=str)
    parser.add_argument('--pi0_config', default='pi0_libero', help='which pi0 config to use', type=str)
    parser.add_argument('--eval_at_begin', default=1, help='whether to evaluate at the beginning of training', type=int)
    parser.add_argument('--max_timesteps', default=500, help='max timesteps per episode', type=int)
    parser.add_argument('--kl_coeff', default=0.0, help='coefficient for KL loss', type=float)
    parser.add_argument('--qwarmup', default=0, help='whether to warmup the Q network', type=int)
    parser.add_argument('--use_res', default=0, help='whether to use residual learner', type=int)
    parser.add_argument('--res_coeff', default=0.1, help='coefficient for residual action', type=float)
    parser.add_argument('--res_H', default=100_000, help='horizon for residual action', type=int)
    parser.add_argument('--td3_noise_scale', default=0.2, help='TD3 noise scale', type=float)
    parser.add_argument('--label', default='', help='label for wandb run', type=str)
    parser.add_argument('--action_magnitude', default=1.0, help='magnitude of actions', type=float)
    parser.add_argument('--decay_kl', default=0, help='whether to decay kl coeff', type=int)
    parser.add_argument('--media_log_fold', default=10, help='fold for media logging', type=int)
    parser.add_argument('--dataset_root', default='', help='root directory for robomimic dataset', type=str)
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        # action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(dict_pretty_str(variant))
    main(variant)
    sys.exit()
    