# 项目说明

## 环境配置

1. **Python 版本**  
   本项目使用 **Python 3.10**。请确保你创建的虚拟环境使用该版本。

2. **安装 PyTorch**  
   根据你的硬件设备选择安装命令：
   
   - **有 GPU（CUDA）**：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     > 将 `cu118` 替换为你显卡对应的 CUDA 版本`。

   - **CPU-only**：
     ```bash
     pip install torch torchvision torchaudio
     ```

3. **安装依赖**  
   使用项目提供的 `requirements.txt` 安装其他依赖：
   ```bash
   pip install -r requirements.txt
   ``` 
   主要下载的库为gymnasium，stable_baselines3，numpy，ale-py，tensorboard等，按照python 3.10下载默认版本即可
如果环境配置有问题可以随时在群里询问

## 项目训练过程记录
    本项目使用swanlab对训练过程各个参数进行记录，相比于tensorboard该软件能更好进行训练曲线对比，参数查看。请创建你的Swanlab账号并且正确使用自己的api登陆，确保能够正确使用swanlab。

## 项目训练指令
     运行TRPO算法的Trajectory和Rollout版本可以分别参考trpo_traj.sh和trpo_rollout.sh

## TRPO Mujoco后台测试

如果想在 `Humanoid-v5`、`HalfCheetah-v5`、`Walker2d-v5` 三个环境下同时测试 rollout 和 trajectory 两种采样方式，可以运行：

```bash
./run_trpo_mujoco_background.sh
```

该脚本会启动 6 个后台任务：

- `Humanoid-v5` + rollout
- `Humanoid-v5` + trajectory
- `HalfCheetah-v5` + rollout
- `HalfCheetah-v5` + trajectory
- `Walker2d-v5` + rollout
- `Walker2d-v5` + trajectory

日志会保存在 `outputs/background_logs/`。项目默认 `use_swanlab: true`，运行前请先完成 SwanLab 登录，确保训练曲线能上传到自己的 SwanLab 项目中。
