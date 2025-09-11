from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from openpi_client import image_tools
import math
import PIL

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None, eval_at_begin=True, dp_unnorm_transform=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    warmup_steps = variant.start_online_updates * variant.query_freq
                            
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            if len(online_replay_buffer) <= variant.start_online_updates:
                res_prob = -1.0
                use_random = True
            else:
                res_prob = (i-warmup_steps)/ variant.res_H if i <= (variant.res_H + warmup_steps) else 1.0
                use_random = False
            traj = collect_traj(variant, agent, env, i, agent_dp, res_prob, dp_unnorm_transform, use_random)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            wandb_logger.log({
                'replay_buffer_size': len(online_replay_buffer),
                'episode_return (exploration)': traj['episode_return'],
                'is_success (exploration)': int(traj['is_success']),
                'residual/res_prob': res_prob,
            }, i)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step
                
            # perform first visualization before updating
            if i == 0 and eval_at_begin:
                print('performing evaluation for initial checkpoint')
                if perform_control_evals:
                    perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp, res_prob, dp_unnorm_transform, use_random)
                if hasattr(agent, 'perform_eval'):
                    agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

            if True:
                for _ in range(num_gradsteps):

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    
                    # noise&&clean space Critic & Actor updates
                    if len(online_replay_buffer) <= variant.start_online_updates:
                        if variant.qwarmup:
                            if variant.use_res:
                                update_info, res_update_info = agent.update_wo_actor(batch, res_prob)
                            else:
                                update_info = agent.update_wo_actor(batch)
                                res_update_info = {}
                                distill_info = agent.distill(batch)
                        else:
                            update_info = {}
                            res_update_info = {}
                    else:
                        if variant.use_res:
                            update_info, res_update_info = agent.update(batch, res_prob)
                        else:
                            # update_info = agent.update(batch)
                            res_update_info = {}               
                            update_info, distill_info = agent.update_with_bc(batch)                             

                    pbar.update()
                    i += 1
                    
                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2 and i%variant.media_log_interval == 0:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                            else:
                                continue
                                
                        res_update_info = {k: jax.device_get(v) for k, v in res_update_info.items()}
                        for k, v in res_update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'residual/{k}': v}, step=i)
                            elif v.ndim <= 2 and i%variant.media_log_interval == 0:
                                wandb_logger.log_histogram(f'residual/{k}', v, i)
                            else:
                                continue
                            
                        distill_info = {k: jax.device_get(v) for k, v in distill_info.items()}
                        for k, v in distill_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'distill/{k}': v}, step=i)
                            elif v.ndim <= 2 and i%variant.media_log_interval == 0:
                                wandb_logger.log_histogram(f'distill/{k}', v, i)
                            else:
                                continue
                            
                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if i % variant.media_log_interval == 0:
                            if perform_control_evals:
                                perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp, res_prob, dp_unnorm_transform, use_random)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0 and i > 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                        replay_buffer.save(f'{variant.outputdir}/replay_buffer_{i}.pkl')

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    norm_actions = np.array(traj['norm_actions']) # (T, chunk_size, action_dim )
    clean_actions = np.array(traj['clean_actions']) # (T, chunk_size, action_dim )
    actual_norm_actions = np.array(traj['actual_norm_actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    distill_noise_actions = np.array(traj['distill_noise']) # (T, chunk_size, action_dim )
    distill_clean_actions = np.array(traj['distill_clean']) # (T, chunk_size, action_dim )

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            norm_actions=norm_actions[t],
            next_norm_actions=norm_actions[t + 1] if t < episode_len - 1 else norm_actions[t],
            clean_actions=clean_actions[t],
            actual_norm_actions=actual_norm_actions[t],
            next_actual_norm_actions=actual_norm_actions[t + 1] if t < episode_len - 1 else actual_norm_actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
            distill_noise_actions=distill_noise_actions[t],
            distill_clean_actions=distill_clean_actions[t],
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, res_prob=0.0, dp_unnorm_transform=None, use_random=False):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    use_res = variant.use_res
    res_coeff = variant.res_coeff
    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
        action_real_dim = 7
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []
    norm_action_list = []
    clean_action_list = []
    actual_norm_action_list = []
    distill_noise_list = []
    distill_clean_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if use_random:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))

                actions_noise = noise[0] # squeeze batch dim
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = actions_noise[None] # add batch dim

            action_dict = agent_dp.infer(obs_pi_zero, noise=noise)
            
            # model_noise = agent_dp.reverse_infer(obs_pi_zero, action=model_actions)["noise"]
            
            # # a.
            actions = action_dict["actions"]

            
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
            norm_action_list.append(action_dict["norm_actions"][:query_frequency])
            clean_action_list.append(action_dict["actions"][:query_frequency])
            actual_norm_action_list.append(action_dict["norm_actions"][:query_frequency])
            
            # for distill only
            rng, key = jax.random.split(rng)
            distill_noise = jax.random.normal(rng, (1, 50, 32))
            
            distill_action_dict = agent_dp.infer(obs_pi_zero, noise=distill_noise)
            
            distill_noise_list.append(distill_noise[0])
            distill_clean_list.append(distill_action_dict["norm_actions"][:query_frequency])
            
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'norm_actions': norm_action_list,
        'clean_actions': clean_action_list,
        'actual_norm_actions': actual_norm_action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'distill_noise': distill_noise_list,
        'distill_clean': distill_clean_list,
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None, res_prob=0.0, dp_unnorm_transform=None, use_random=False):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    use_res = variant.use_res
    res_coeff = variant.res_coeff
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
            action_real_dim = 7
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                
                
                if use_random:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = actions_noise[None]
                    
                action_dict = agent_dp.infer(obs_pi_zero, noise=noise)

                # # a.
                actions = action_dict["actions"]
                
                # b.

            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list)[::2].transpose(0, 3, 1, 2) # downsample by 2 and change to (T, C, H, W) format
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=30, format='mp4')}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
