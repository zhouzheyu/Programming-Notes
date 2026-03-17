# argparse

## 核心用法

让 Python 脚本支持像这样运行：

```bash
python train.py --lr 0.001 --batch_size 32 --epochs 10
```

使用 argparse 基本就三步：

```python
import argparse

# 1. 创建解析器
parser = argparse.ArgumentParser()

# 2. 添加参数
parser.add_argument('--lr', type=float, default=0.001)

# 3. 解析参数
args = parser.parse_args()
```

### add_argument函数的常见参数

name:

```python
parser.add_argument('--lr')
```

type:

```python
parser.add_argument('--lr', type=float)
```

 default：默认值

```python
parser.add_argument('--lr', default=0.001)
```

help：说明

```python
parser.add_argument('--lr', help='学习率')
```

required：是否必须，不传会报错

```python
parser.add_argument('--input', required=True)
```

choices：取值范围，防止乱传参数

```python
parser.add_argument('--model', choices=['xgb', 'lgb', 'logit'])
```

action：行为控制。

action不和default连用，否则永远都是true。

```python
# store: 存值（默认，就算不写action一样可以存起来）
parser.add_argument('--lr', action='store')
# store_true: 开关，bash指令中有这个参数就是True，没有就是False
# 运行 python train.py --use_gpu，结果 args.use_gpu = True
# 如果不写 python train.py，结果 args.use_gpu = False
parser.add_argument('--use_gpu', action='store_true')
# store_false
parser.add_argument('--no_cuda', action='store_false')
```



