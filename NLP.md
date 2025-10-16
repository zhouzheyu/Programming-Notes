# transformers

huggingface：https://huggingface.co/

官方文档：https://huggingface.co/docs/transformers

gitHub:https://github.com/huggingface/transformers

## tokenizer

用于对词序列进行编码和解码。

### 创建 tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",   # 模型名或路径)
```

### 调用 tokenizer

如果在tokenizer阶段就指定padding=True，那么就会在后续的处理中视为正常单词，浪费内存。

因此一般指定padding=False，后续使用DataCollatorWithPaddin进行数据填充。

```python
tokens = tokenizer(
    ["Hello world", "Hi"], # 输入：单个字符串或字符串列表
    add_special_tokens=True,   # 是否加 [CLS]/[SEP] 等特殊符号
    max_length=20,             # 最大长度
    padding=False,              # 是否 padding（True / 'max_length' / 'longest'）
    truncation=False,           # 是否截断（True 或 'longest_first'）
    return_tensors=None,        # 'pt' / 'tf' / 'np'
    return_token_type_ids=True, # 是否返回 token_type_ids（句对任务用）
    return_attention_mask=True, # 是否返回 attention_mask
    is_split_into_words=False,  # 输入是否是已经分好的词（而不是原始句子）
    stride=0,                   # 滑动窗口步长（长文本切分）
    return_overflowing_tokens=False, # 是否返回溢出的切片
    return_offsets_mapping=False,    # 返回字符到 token 的位置映射
)
```

### 返回值

```python
{
    'input_ids': [[101, 7592, 2088, 102], [101, 7632, 102]],
    'token_type_ids': [[0, 0, 0, 0], [0, 0, 0]], # 有些模型可能没有这个字段，比如 GPT 类模型
    'attention_mask': [[1, 1, 1, 1], [1, 1, 1]] # 1告诉模型哪些位置是实际内容，哪些位置是padding
}
```

可以通过索引得到相应的token：

```
embedding = tokens["input_ids"][0]
```

### 编码+解码

```python
# 编码
enc = tokenizer("Hello world", return_tensors="pt", padding="max_length", max_length=10)
# 单句解码
text = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
# 批次解码
text = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)
```

## models

### 模型定义和调用

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 输入文本 → token → 模型预测
inputs = tokenizer("I love Hugging Face!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

model.generate：结合tokenizer.decode做生成，让output变成词语

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 编码
inputs = tokenizer("Hello, Hugging Face", return_tensors="pt")

# 推理生成
outputs = model.generate(**inputs, max_length=20)

# 解码成文本
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### 扩展现有模型





## TrainingArguments



## Trainer



## DataCollatorWithPadding

用于数据填充：

* 接收一个 batch（list of dicts），每个 dict 是 tokenizer 编码后的结果（`input_ids`, `attention_mask` 等）。
* 找到 batch 中最长的序列长度（或按你设置的 `max_length`），对所有样本用 tokenizer 的 pad 方法补齐。
* 返回 PyTorch 或 TensorFlow tensor（取决于你设置的 `return_tensors`）。

常用参数：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer,                # 必填，负责pad逻辑的tokenizer
    padding=True,              # True=动态padding, 也可传"longest"/"max_length"
    max_length=None,           # 指定固定填充到的长度
    pad_to_multiple_of=None,   # 使长度补到某个倍数（适配硬件加速）
    return_tensors="pt"        # 输出类型："pt"=PyTorch, "tf"=TensorFlow, "np"=NumPy
)
```

在 Trainer中使用：Trainer内部会自动创建 DataLoader，只要把data_collator传进去就行

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,   # 直接传进去
    tokenizer=tokenizer
)
```

