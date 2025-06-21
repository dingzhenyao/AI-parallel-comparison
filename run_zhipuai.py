from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch  # 需要补充导入torch模块

# 修改为ChatGLM3-6B的本地路径
model_name = "/mnt/data/chatglm3-6b" 

prompt = "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？"  #根据需要切换问题

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",  # 自动选择float32/float16
    device_map="auto"    # 自动分配GPU/CPU资源
).eval()

# 生成输入tensor并移动到模型所在设备
inputs = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids

streamer = TextStreamer(tokenizer)
outputs = model.generate(
    inputs, 
    streamer=streamer,
    max_new_tokens=500,
    temperature=0.8,     # 推荐添加生成参数
    top_p=0.9            # 推荐添加生成参数
)