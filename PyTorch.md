# Pytorch

一般的训练代码编写套路：

```python
def train(model, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
  best_dev_UAS = 0 # 指标的最高测试结果
  optimizer = optim.Adam(model.parameters(), lr = lr)
  loss_func = nn.CrossEntropyLoss()
  for epoch in range(n_epochs):
    print("Epoch {:} out of {:}".format(epoch + 1, n_epochs)) # 第epoch轮
  	dev_UAS = train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size)
    if dev_UAS > best_dev_UAS:
    	best_dev_UAS = dev_UAS # 更新最高测试指标
    	torch.save(parser.model.state_dict(), output_path)
    print("")
    
def train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size):
  # 训练模式
  model.train()
  n_minibatches = math.ceil(len(train_data) / batch_size)
  loss_meter = AverageMeter()
  with tqdm(total=(n_minibatches)) as prog:
    # 每来一个batch训练一次，并更新loss
    for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
      optimizer.zero_grad()   # remove any baggage in the optimizer
      train_x = torch.from_numpy(train_x).long()
      train_y = torch.from_numpy(train_y.nonzero()[1]).long()
      y_hat = parser.model(train_x)
      loss = loss_func(y_hat, train_y)
      loss.backward()
      optimizer.step()
      ### END YOUR CODE
      prog.update(1)
      loss_meter.update(loss.item())
  # 在所有训练样本上的平均损失
  print ("Average Train Loss: {}".format(loss_meter.avg))
  # 评估模式，关闭Dropout，评估指标。
  print("Evaluating on dev set",)
  model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
  dev_UAS, _ = parser.parse(dev_data)
  print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
  return dev_UAS # 返回测试结果
```

## einops

### rearrange

```python
from einops import rearrange
# 示例1：改变维度顺序
x = torch.randn(2, 3, 4)  # shape: [batch, height, width]
y = rearrange(x, 'b h w -> b w h')
print(y.shape)  # [2, 4, 3]
# 示例2：把 (h, w) 展平为一个维度
y = rearrange(x, 'b h w -> b (h w)')
print(y.shape)  # [2, 12]
# 示例3：分块（比如把图像拆成patch）
x = torch.randn(1, 3, 32, 32)  # [batch, channel, height, width]
patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
print(patches.shape)  # [1, 64, 48]
# 示例4:扩展维度
y = rearrange(x, 'b h w -> 1 b h w')
print(y.shape) # [1, 2, 3, 4]
```

### reduce

支持sum, mean, max, min, prod（连乘）

```python
from einops import reduce

x = torch.randn(10, 3, 32, 32)  # [batch, channel, h, w]

# 示例1：在空间维度上求平均
y = reduce(x, 'b c h w -> b c', 'mean')
print(y.shape)  # [10, 3]
y = reduce(x, 'b c h w -> c h w', 'sum')
print(y.shape)  # [3, 32, 32]
```

### repeat

```python
# 扩展新维度并复制
x = torch.randn(10, 64)  # [batch, dim]
y = repeat(x, 'b d -> b t d', t=5)
print(y.shape)  # [10, 5, 64]
# 广播向量以匹配矩阵形状
token = torch.randn(1, 512)
seq = repeat(token, '1 d -> n d', n=128)
print(seq.shape)  # [128, 512]
# 在已有维度上复制
x = torch.randn(1, 3, 32, 32)
y = repeat(x, 'b c h w -> (b n) c h w', n=8)
print(y.shape)  # [8, 3, 32, 32]
```

### torch.einsum

任意维度的线代乘法求和，但不支持einpos的维度变换。

```python
x = torch.ones(2, 3, 4)
y = torch.ones(2, 3, 4)
z = torch.einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

## torch

### 杂七杂八

指定GPU为mac版本：

```python
# 选择设备，mac的设备是mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 确保输入数据 imgs 在 MPS 上
imgs = imgs.to(device)
# ✅ 确保 targets 也在 MPS
targets = targets.to(device)
# 初始化模型并移动到 MPS
model = SimpleNN().to(device)
```

requires_grad=True: 指定参数可求导。

大多数场景都默认requires_grad=True：nn.Parameter(torch.randn(xx, xx))。

需要手动指定的场景：

* 迁移学习、微调、LoRA冻结部分层
* 
* 让输入也参与梯度
* 临时关闭梯度计算（推理/评估）
* detach() 手动切断梯度流

```python
# 对于张量需要手动定义
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # tensor([7.])
# 模型参数默认带梯度
for name, param in model.named_parameters():
    print(name, param.requires_grad)
# 冻结参数
for param in model.encoder.parameters():
    param.requires_grad = False
# 推理阶段
with torch.no_grad():
    y_pred = model(x)
# detach() 手动切断梯度流
z = encoder(x).detach()  # 不让 encoder 部分反向传播
```

torch.load(filepath): 加载

```python
# 加载对应的张量
loaded_tensor = torch.load('tensor.pt')
# 加载到模型参数
model.load_state_dict(torch.load('model_weights.pth'))
# 加载一个字典
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

torch.save(obj, filepath): 保存

```python
# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')
# 保存字典对象
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

tensor和numpy的相互转换：

```python
isinstance(obj, torch.Tensor)  # 检查是否为 Tensor
isinstance(obj, np.ndarray)    # 检查是否为 ndarray
tensor.numpy()  # 转换为 numpy 数组
torch.from_numpy(ndarray)  # 转换为 Tensor
```

"=="判断：
```python
X1 = torch.tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype = torch.float32)

X2 = torch.tensor([[2, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype = torch.float32)

print(X1 == X2)

# tensor([[False,  True,  True,  True],
#         [ True,  True,  True,  True],
#         [ True,  True,  True,  True]])
```

torch.argmax():
```python
# 找到张量中最大值的索引id
max_id = torch.argmax(X)
# 找到某一维上的最大值索引，返回的是一个tensor序列
max_ids = torch.argmax(X, dim = -1)
```

torch.where(cond, a, b)：当cond成立时选择a中的对应元素，否则选择b中的对应元素。

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])
cond = torch.tensor([True, False, True])

result = torch.where(cond, a, b)
print(result)  # tensor([ 1, 20,  3])

# 这里用了广播
labels = torch.where(labels == 8505, torch.tensor(1), torch.tensor(0))
```


归一化函数：
```python
import torch.nn.function as F
# dim：操作的维度，在哪一维进行归一化
# p：范数阶数
X_phi = F.normalize(input = X, dim = -1, p = 2)
```


torch.linspace：生成n个从x_min到x_max的一维张量：
```python
X = torch.linspace(x_min, x_max, n)
```

torch.split(tensor, split_size:[], dim)，张量的分割
```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

temp1, temp2 = torch.split(tensor, [1, 2], dim = -1)

print(temp1, temp2)
```

数据类型的转换：Y = X.long()
```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(tensor)

tensor = tensor.long()
print(tensor)
```

求转秩：

```python
Y = X.t()
```

### 维度操作

X.view()：改变维度大小

```python
# 如果 x 是通过 permute 或 transpose 得到的（即不连续），需要先 .contiguous()
x = torch.randn(2, 3, 4)
y = x.view(6, 4)  # [2*3, 4]
```

X.reshape()：改变维度大小

```python
# 自动处理，必要时复制内存，不必连续
x = torch.randn(2, 3, 4)
y = x.reshape(6, 4)
```

X.permute()：改变维度顺序

```python
X = X.permute(0, 3, 1, 2)
```

X.squeeze(dim)：删除大小为1的维度

```python
# 删除大小为1的维度
Y1 = X.squeeze() # 去掉所有大小为删除所有维度大小为1的维度。
Y2 = X.squeeze(-1) # 若最后一维的大小为1，则删除
```

X.unsqueeze(dim)：在指定维度增广维度1

```python
# arr.shape =torch.Size([3, 4])
arr = torch.tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

arr_unsqueezed = arr.unsqueeze(0)
# arr_unsqueezed.shape =torch.Size([1, 3, 4])

arr_unsqueezed = arr.unsqueeze(1)
# arr_unsqueezed.shape =torch.Size([3, 1, 4])
```

torch.cat([A, B], dim)：张量的拼接

torch.stack([A, B], dim)：张量的堆叠

### 数据操作

torch.zeros_like()：

```python
# 生成和X相同形状的全零张量
X = torch.rand(3, 4)
Y = torch.zeros(X)
```

torch.sum(Tensor, dim), torch.mean(Tensor, dim):

```python
# dim参数代表对哪一维操作。
# sum的后面可以跟squeeze()
Y = torch.sum(X, dim = -1)
Y = torch.sum(X, dim = -1).squeeze()
# torch.mean同理
Y = torch.mean(X, dim = -1)
```

torch.bmm(tensor1, tensor2)：批量矩阵乘法，常和X.squeeze和X.unsqueeze连用

```python
# dec_hidden: (b, h), enc_hiddens_proj: (b, src_len, h)
# (b, h), (b, src_len, h) -> (b, 1, h), (b, h, src_len)，相乘得到(b, 1, src_len)，最后squeeze成(b, src_len)
e_t = torch.bmm(dec_hidden.unsqueeze(1),  enc_hiddens_proj.transpose(1, 2)).squeeze(1)
```

torch.matmul(A, B)：矩阵乘法

```python
# 当输入是大于二维的张量，torch.matmul(A, B) 的行为是：将最后两个维度当作矩阵进行乘法，其余维度按广播规则进行广播。
A: [b, m, k]
B: [b, k, n]
C = torch.matmul(A, B) → [b, m, n]
```

## torch.nn

### nn.Module

#### self.named_parameters()

返回字典，对应模型中的所有变量名和数值

冻结指定参数，不参与更新：

```python
for name, param in self.named_parameters():
    if ... : # 是指定的模块
        param.requires_grad = False
```

#### model.train()

切换到训练模式

#### model.eval()

切换到评估模式

### nn.Embedding

nn.Embedding(num_embeddings, embedding_dim, padding_idx)：初始化所有词的embedding

```python
# 为了保证所有句子一样长，通常会用连续的'<pad>'填充，所有'<pad>'共用一个padding_idx
# vocab.src可能长这样：src = {'<pad>': 0, 'hello': 1, 'world': 2}
src_pad_token_idx = vocab.src['<pad>']
tgt_pad_token_idx = vocab.tgt['<pad>']
# 初始化Embedding
source = nn.Embedding(len(vocab.src), embedding_dim=self.embed_size, padding_idx=src_pad_token_idx)
target = nn.Embedding(len(vocab.tgt), embedding_dim=self.embed_size, padding_idx=tgt_pad_token_idx)
# 通过索引得到对应idx的embedding
input = torch.tensor([1, 2])  # 相当于输入 'hello', 'world'
output = source(input) # torch.Size([2, 5])
# 得到多个batch的embedding
input = torch.tensor([[1, 2, 0], [2, 1, 0]])  # batch size = 2，句长 = 3
output = source(input) # torch.Size([2, 3, 5])
```

### nn.functional

损失函数的定义和调用：

```python
import torch.nn.functional as F

loss = F.cross_entropy(logits, labels, reduction='mean')
```

## torch.utils.data

数据集类的定义和调用：

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, data, args):
        self.data = data          # 这里可以是numpy数组、列表、torch张量等
        ... = args                # 根据参数指定变量

    def __len__(self):
        return len(self.data)     # 返回数据集样本数

    def __getitem__(self, idx):
        x = self.data[idx]        # 根据索引取出数据
        y = self.labels[idx]      # 取出对应标签
        return x, y

if __name__ == '__main__':
	dataset = MyDataset(my_data, args)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	for batch_x, batch_y in dataloader:
		...
```

如果不写dataloader的话，那么就是一次循环取一个样本：

```python
for x, y in dataset:
	...
```

## torch.utils.tensorboard

SummaryWriter:

```python
from torch.utils.tensorboard import SummaryWriter

# 指定保存路径
tensorboard_path = "nmt" if args['--cuda'] else "nmt_local"
writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

train_iter = 0
for epoch in max_epochs:
  for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
    train_iter += 1
    # 每迭代多少次打印一次日志
    if train_iter % log_every == 0:
      # add_scalar的三个参数：标签名、横坐标、纵坐标。标签名不同，图也不同。
      writer.add_scalar("loss/train", report_loss / report_tgt_words, train_iter)
      writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), train_iter)
      # add_text的三个参数：标签名、文本内容、当前的训练步数
    	writer.add_text(tag='info', text_string='Training started!', global_step=0)
```

查看方法：

* run the following in a separate terminal in which you have also activated the cs224n-cpu conda environment
* 从而在网址http://localhost:6006/即可查看

```bash
tensorboard −−logdir = runs −−port 6006
```









