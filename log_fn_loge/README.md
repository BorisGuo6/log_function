# 安装torch

https://pytorch.org/get-started/previous-versions/
conda activate py3.9-log-fn

# 安装其他依赖

pip install -r requirements.txt
gym==0.25.2

# 查看服务器剩余资源

nvidia-smi

# 查看服务器上所有正在跑的程序

htop

3000

bash /home2/ad/liuqi/log_function/log_actor_critic.sh
tensorboard --logdir=/home2/ad/liuqi/log_function/tensorboard_log --port=6007
