# AI-parallel-comparison
# 大模型本地推理环境配置指南

## 环境要求
- ModelScope 平台账号（已绑定阿里云）
- CPU 计算资源（平台提供）
- Linux 环境（推荐 Ubuntu 20.04+）

> **重要提示**：
> 1. CPU 资源1小时未使用会自动释放，环境会被清空
> 2. 一次只下载一个大模型，避免存储不足（每个模型约需14-16GB空间）

---

## 一、注册 ModelScope
1. 访问 [ModelScope 官网](https://www.modelscope.cn/home)
2. 右上角完成新用户注册
3. 登录后绑定阿里云账号
4. 获取免费云计算资源

---

## 二、环境配置

### 1. 启动 Notebook 终端
在 ModelScope 控制台：
- 点击 Terminal 图标打开命令行环境

### 2. Conda 环境安装（如未预装）
```bash
cd /opt/conda/envs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
```

### 3. 创建并激活 Python 环境
```bash
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```

---

## 三、安装依赖

### 1. 基础框架
```bash
pip install \
torch==2.3.0+cpu \
torchvision==0.18.0+cpu \
--index-url https://download.pytorch.org/whl/cpu
```

### 2. 核心依赖
```bash
pip install -U pip setuptools wheel
pip install \
"intel-extension-for-transformers==1.4.2" \
"neural-compressor==2.5" \
"transformers==4.33.3" \
"modelscope==1.9.5" \
"pydantic==1.10.13" \
"sentencepiece" \
"tiktoken" \
"einops" \
"transformers_stream_generator" \
"uvicorn" \
"fastapi" \
"yacs" \
"setuptools_scm"
```

### 3. 可选增强组件
```bash
pip install tqdm huggingface-hub
pip install fschat --use-pep517
```

---

## 四、下载大模型

### 1. 切换到数据目录
```bash
cd /mnt/data
```

### 2. 下载模型（任选其一）
```bash
# 通义千问
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git

# 清华智谱
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git

```

> **存储建议**：  
> 每个模型约需14-16GB空间，建议一次只下载一个模型

---


## 常见问题解决
1. **conda 命令未找到**  
   重新执行：  
   `source /opt/conda/etc/profile.d/conda.sh`

2. **存储空间不足**  
   - 检查当前下载的模型数量
   - 清理不需要的文件：  
     `rm -rf /mnt/data/*`

3. **依赖安装失败**  
   尝试指定国内镜像源：  
   `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package]`

---

## 参考资源
1. [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers)
2. [Transformers 官方文档](https://huggingface.co/docs/transformers)
3. [ModelScope 模型库](https://www.modelscope.cn/models)

> 环境有效期：CPU资源1小时未使用会自动释放，请及时保存重要数据
