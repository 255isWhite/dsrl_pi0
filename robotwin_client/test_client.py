import zmq
import pickle
import time
import numpy as np

def main():
    host = "localhost"
    port = 21451
    address = f"tcp://{host}:{port}"

    # 建立 ZeroMQ REQ socket
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(address)
    print(f"[Client] Connected to {address}")

    # 先 reset 一下环境
    socket.send(pickle.dumps({"command": "reset"}))
    response = pickle.loads(socket.recv())
    print(f"[Client] Reset response keys={list(response.keys())}")

    # 这里假设 action 维度是 7，可以根据实际调整
    action_dim = 14
    zero_action = np.zeros(action_dim, dtype=np.float32)
    zero_action[6] = 1.0  # gripper open
    zero_action[0] = 0.01

    # 循环不断发送 step
    step_count = 0
    while True:
        data = {
            "command": "step",
            "action": zero_action
        }
        socket.send(pickle.dumps(data))
        response = pickle.loads(socket.recv())
        step_count += 1
        print(f"[Client] Step {step_count}, got keys={list(response.keys())}, done={response.get('done')}")

        time.sleep(0.1)  # 控制请求频率，避免刷爆 server

if __name__ == "__main__":
    main()
