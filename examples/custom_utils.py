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

class RoboTwinEnv:
    @run_in_robotwin
    def __init__(self, task_name: str, **kwargs):
        self.env, self.env_config = create_robotwin_env(task_name)
        self.save_dir = kwargs.get("save_dir", None)
        current_time = get_beijing_time()
        save_dir = Path(f"{self.save_dir}/{current_time}/log")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        video_dir = Path(f"{self.save_dir}/{current_time}/video")
        from robotwin import task_config
        camera_config = task_config.load("_camera_config")
        camera_config = camera_config[self.env_config["camera"]["head_camera_type"]]
        
        self.video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_dir.mkdir(parents=True, exist_ok=True)
        self.env_config['eval_video_save_dir'] = video_dir

        self.env.setup_demo(**self.env_config)
        self.episode_info = self.env.play_once()
        self.episode_info_list = [self.episode_info["info"]]
        self.task_name = task_name
        self.set_video_ffmpeg()
        print(f"RoboTwinEnv initialized with task: {task_name}")
        
        self.start_time = time.time()
    
        
    def set_video_ffmpeg(self):
        current_time = get_beijing_time()
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                self.video_size,
                "-framerate",
                "10",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "28",
                f"{self.env.eval_video_path}/{self.task_name}_{current_time}.mp4",
            ],
            stdin=subprocess.PIPE,
        )
        self.env._set_eval_video_ffmpeg(ffmpeg)

    def step(self, action):
        self.env.take_action(action)
        done = self.env.eval_success
        # obs = self.env.get_obs()
        return None, done
    
    def get_obs(self):
        return self.env.get_obs()

    @run_in_robotwin
    def reset(self):        
        print(f"millisecons to last reset: {(time.time() - self.start_time) * 1000:.2f} ms")
        self.start_time = time.time()
        
        
        
        results = generate_episode_descriptions(self.task_name, self.episode_info_list, 1)
        instruction = np.random.choice(results[0]["unseen"])
        self.env.set_instruction(instruction=instruction)  # set language instruction
        self.env.setup_demo(**self.env_config)
        self.env._del_eval_video_ffmpeg()
        self.set_video_ffmpeg()  # reset video ffmpeg
        obs = self.env.get_obs()
        
        
        print(f"reset spend time: {(time.time() - self.start_time) * 1000:.2f} ms")
        
        return obs, instruction
