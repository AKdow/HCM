# 打包阶段
```
# 1. 确保已装好编译依赖（仅首次）
conda activate hrmpy311
pip install ninja wheel build setuptools-scm

# 2. 一次性生成 wheel（–no-deps 防止把 torch 等巨型依赖也打进来）
pip wheel flash-attn==2.8.3 \
        --no-deps \
        --wheel-dir ./flash_attn_whl

# 3. 检查产物
ls -lh flash_attn_whl/
# 示例输出
# flash_attn-2.8.3+cu121torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

# 安装
```
# 1. 新建/激活目标环境（Python 版本需与 wheel 标签一致，这里是 3.11）
conda create -n new_env python=3.11 -y
conda activate new_env

# 2. 先装运行时依赖（torch 必须与你编译时 CUDA 版本对应）
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# 如有需要再装 einops
pip install einops

# 3. 离线安装打好的 flash-attn（--no-index 防止又去 PyPI）
pip install --no-index --find-links /your_group/shared/wheels/flash_attn/ flash-attn==2.8.3

# 4. 验证
python -c "import flash_attn, torch, platform, sys;\
           print('flash_attn:', flash_attn.__version__);\
           print('torch:', torch.__version__);\
           print('cuda:', torch.version.cuda);\
           print('python:', platform.python_version())"
```
