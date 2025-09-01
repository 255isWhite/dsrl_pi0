def dict_pretty_str(d: dict) -> str:
    max_key_len = max(len(str(k)) for k in d.keys())
    lines = [f"{k:<{max_key_len}} : {v}" for k, v in d.items()]
    return "\n".join(lines)

import hashlib

def safe_group_name(name: str, max_len: int = 120) -> str:
    """确保 group_name 不超过指定长度，并避免截断导致的重复"""
    if len(name) <= max_len:
        return name
    # 保留前 max_len-9 字符 + '_' + 8位hash
    prefix = name[:max_len - 9]
    suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{prefix}_{suffix}"

import os
import sys
from pathlib import Path
import robotwin
import yaml
import subprocess
from datetime import datetime
from datetime import timedelta
import pytz
import time
import zmq
import pickle

def get_beijing_time():
    beijing_tz = pytz.timezone('Asia/Shanghai') 
    now_utc = datetime.now().replace(tzinfo=pytz.utc)
    now_beijing = now_utc.astimezone(beijing_tz)
    current_time = now_beijing.strftime('%m_%d_%H_%M')
    return current_time

B_ROOT = Path(robotwin.__file__).parent

def run_in_robotwin(func):
    def wrapper(*env_config, **kwenv_config):
        old_cwd = os.getcwd()
        sys_path_backup = list(sys.path)
        try:
            # 切换到 B 的根目录
            os.chdir(B_ROOT)

            # 把 robotwin/envs 临时加到 sys.path，解决 "from envs.xxx import ..."
            sys.path.insert(0, str(B_ROOT / "envs"))

            print(f"[run_in_robotwin] cwd switched to: {os.getcwd()}")
            return func(*env_config, **kwenv_config)
        finally:
            # 恢复环境
            os.chdir(old_cwd)
            sys.path = sys_path_backup
            print(f"[run_in_robotwin] cwd restored to: {os.getcwd()}")
    return wrapper

@run_in_robotwin
def create_robotwin_env(task_name: str, **kwenv_config):
        print("Now in RoboTwin dir:", os.getcwd())
        from robotwin import task_config
        env_config = task_config.load("demo_clean")
        from robotwin.envs import CONFIGS_PATH
        embodiment_type = env_config.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = env_config["camera"]["head_camera_type"]
        env_config["head_camera_h"] = _camera_config[head_camera_type]["h"]
        env_config["head_camera_w"] = _camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            env_config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            env_config["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            env_config["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            env_config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            env_config["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            env_config["embodiment_dis"] = embodiment_type[2]
            env_config["dual_arm_embodied"] = False
        else:
            raise "embodiment items should be 1 or 3"
        
        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_env_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_env_config

        env_config["left_embodiment_config"] = get_embodiment_config(env_config["left_robot_file"])
        env_config["right_embodiment_config"] = get_embodiment_config(env_config["right_robot_file"])

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
            
        def class_decorator(task_name):
            envs_module = importlib.import_module(f"envs.{task_name}")
            try:
                env_class = getattr(envs_module, task_name)
                env_instance = env_class()
            except:
                raise SystemExit("No Task")
            return env_instance
        env = getattr(__import__(f"robotwin.envs.{task_name}",
            fromlist=[task_name]), task_name)()
        
        return env, env_config
    
import numpy as np
import importlib

@run_in_robotwin
def import_generate_episode_descriptions():
    module = importlib.import_module(
        "robotwin.description.utils.generate_episode_instructions"
    )
    return module.generate_episode_descriptions

generate_episode_descriptions = import_generate_episode_descriptions()


    
# ============ Client ==============
class ClientEnv:
    def __init__(self, full_address: str, **kwargs):
        self.full_address = full_address
        host, port = full_address.split(':')
        port = int(port)
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[Client] Connected to {full_address}")

    def _print_obs(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"    [Obs] {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, dict):
                    print(f"    [Obs] {k}: (dict)")
                    self._print_obs(v)  # 递归打印
                else:
                    print(f"    [Obs] {k}: type={type(v)}, value={str(v)[:50]}")
        else:
            print(f"    [Obs] type={type(obs)}, value={str(obs)[:50]}")

    def step(self, action):
        fake_message = {'command': 'step', 'action': action}
        t0 = time.time() * 1000
        # print(f"[Client] Sending step action len={len(action)}")
        self.socket.send(pickle.dumps(fake_message))
        message = self.socket.recv()
        t1 = time.time() * 1000
        data = pickle.loads(message)
        # print(f"[Client] Got step reply keys={list(data.keys())}, 往返耗时={t1 - t0:.2f} ms")
        # if "obs" in data:
        #     self._print_obs(data["obs"])
        done = data.get('done')
        return data.get("obs"), done

    def get_obs(self):
        fake_message = {'command': 'get_obs'}
        t0 = time.time() * 1000
        # print(f"[Client] Sending get_obs")
        self.socket.send(pickle.dumps(fake_message))
        message = self.socket.recv()
        t1 = time.time() * 1000
        data = pickle.loads(message)
        obs = data.get('obs')
        # print(f"[Client] Got obs, 往返耗时={t1 - t0:.2f} ms")
        # self._print_obs(obs)
        return obs

    def reset(self):
        fake_message = {'command': 'reset'}
        t0 = time.time() * 1000
        # print(f"[Client] Sending reset")
        self.socket.send(pickle.dumps(fake_message))
        message = self.socket.recv()
        t1 = time.time() * 1000
        data = pickle.loads(message)
        obs = data.get('obs')
        instruction = data.get('instruction')
        # print(f"[Client] Reset reply: instruction={instruction}, 往返耗时={t1 - t0:.2f} ms")
        # self._print_obs(obs)
        return obs, instruction