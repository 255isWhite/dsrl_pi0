import zmq
import pickle
import argparse
import os
import threading
import time
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
import numpy as np
import importlib
import zmq
import pickle
import cv2
import json
import torch
import random
from collections import deque
B_ROOT = Path(robotwin.__file__).parent

def get_safe_timestamp():
	return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"

def get_beijing_time():
	# beijing_tz = pytz.timezone('Asia/Shanghai') 
	now_utc = datetime.now().replace(tzinfo=pytz.utc)
	# now_beijing = now_utc.astimezone(beijing_tz)
	current_time = now_utc.strftime('%m_%d_%H_%M_%S_%f')
	return current_time

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
def import_generate_episode_descriptions():
	module = importlib.import_module(
		"robotwin.description.utils.generate_episode_instructions"
	)
	return module.generate_episode_descriptions

generate_episode_descriptions = import_generate_episode_descriptions()

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
	
class RoboTwinEnv:
	@run_in_robotwin
	def __init__(self, task_name: str, **kwargs):
		self.env, self.env_config = create_robotwin_env(task_name)
  
		current_time = get_safe_timestamp()
		self.save_dir = kwargs.get("save_dir", "./robotwin_log")
		self.save_json = Path(f"{self.save_dir}/{current_time}/result.json")
		video_dir = Path(f"{self.save_dir}/{current_time}/video")
		video_dir.mkdir(parents=True, exist_ok=True)

		from robotwin import task_config
		camera_config = task_config.load("_camera_config")
		camera_config = camera_config[self.env_config["camera"]["head_camera_type"]]
		
		self.video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
		self.env_config['eval_video_save_dir'] = video_dir

		self.env.setup_demo(**self.env_config)
		self.episode_info = self.env.play_once()
		self.episode_info_list = [self.episode_info["info"]]
		self.task_name = task_name
		self.set_video_ffmpeg()
		print(f"RoboTwinEnv initialized with task: {task_name}")
		
		self.start_time = time.time()
		self.current_success = 0.0
		self.real_ep_count = 0
		self.current_video_name = None
		self.recent_results = deque(maxlen=10)
	
		
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
		self.current_video_name = f"{self.env.eval_video_path}/{self.task_name}_{current_time}.mp4"

	def step(self, action):
		self.env.take_action(action)
		done = self.env.eval_success
		self.current_success = done
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
	
		# 保存上一个episode结果
		if self.real_ep_count > 0:
			success_flag = bool(self.current_success)
			new_name = self.current_video_name.replace(".mp4", "_success.mp4" if success_flag else "_fail.mp4")
			os.rename(self.current_video_name, new_name)

			# 更新统计
			self.recent_results.append(1 if success_flag else 0)

			# 写一行json
			result_line = {
				"task": self.task_name,
				"episode": self.real_ep_count,
				"success": success_flag,
				"window_avg_success_rate": 100.0 * sum(self.recent_results) / len(self.recent_results),
			}
			with open(self.save_json, "a") as f:
				f.write(json.dumps(result_line, ensure_ascii=False) + "\n")

				
		self.set_video_ffmpeg()  # reset video ffmpeg
		obs = self.env.get_obs()
		print(f"reset spend time: {(time.time() - self.start_time) * 1000:.2f} ms")
		
		self.real_ep_count += 1
		return obs, instruction
	

# ============ Server ==============
class ClientModel:
	def __init__(self, host='localhost', port=8080, env=None):
		self.host = host
		self.port = port
		self.env = env
		ctx = zmq.Context()
		self.socket = ctx.socket(zmq.REP)
		self.socket.bind(f"tcp://{self.host}:{self.port}")

	def run(self):
		print(f"[Server] Started at tcp://{self.host}:{self.port}")
		while True:
			message = self.socket.recv()
			data = pickle.loads(message)
			command = data.get('command')
			print(f"[Server] Received command={command}, keys={list(data.keys())}")

			t0 = time.time() * 1000  # ms

			if command == 'reset':
				t1 = time.time() * 1000
				obs, instruction = self.env.reset()
				t2 = time.time() * 1000
				print(f"[Server] env.reset()耗时 {t2 - t1:.2f} ms")
				response = {'obs': obs, 'instruction': instruction}

			elif command == 'step':
				action = data.get('action')
				t1 = time.time() * 1000
				obs, done = self.env.step(action)
				t2 = time.time() * 1000
				print(f"[Server] env.step()耗时 {t2 - t1:.2f} ms")
				response = {'obs': obs, 'done': done}

			elif command == 'get_obs':
				t1 = time.time() * 1000
				obs = self.env.get_obs()
				t2 = time.time() * 1000
				print(f"[Server] env.get_obs()耗时 {t2 - t1:.2f} ms")
				response = {'obs': obs}

			else:
				response = {'error': 'Unknown command'}

			t3 = time.time() * 1000
			print(f"[Server] Responding keys={list(response.keys())}, "
				  f"types={[type(v) for v in response.values()]}, 总耗时={t3 - t0:.2f} ms")
			self.socket.send(pickle.dumps(response))

# ============ Main ==============
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', type=str, default='localhost')
	parser.add_argument('--port', type=int, default=8080)
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--task_name', type=str, required=True)
	parser.add_argument('--save_dir', type=str, default='./robotwin_log')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()
	print(f"gpu id: {args.gpu_id}")

	full_address = f"{args.host}:{args.port}"
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
	print(f"[Main] Start Client Using GPU ID: {args.gpu_id}")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	print(f"[Main] Using Seed: {args.seed}")
	# ====== 创建真实环境 ======
	env = RoboTwinEnv(args.task_name, save_dir=args.save_dir)
	obs, instruction = env.reset()
	print("[Main] Initial Instruction:", instruction)

	# ====== 启动 server 线程 ======
	server = ClientModel(host=args.host, port=args.port, env=env)
	server_thread = threading.Thread(target=server.run, daemon=True)
	server_thread.start()

	time.sleep(1)  # 等 server bind
	
	print("[Main] Env should be ready now.")
  

	# 阻塞主线程，避免退出
	try:
		while True:
			print("Client running... Press Ctrl+C to stop.")
			time.sleep(60)
	except KeyboardInterrupt:
		print("Shutting down client...")