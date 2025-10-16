# Python基础

https://liaoxuefeng.com/books/python/introduction/index.html

学到了什么是python特有的？

切片、迭代、列表推导式、generator生成器、函数式编程（map reduce filter sorted lambda decorator partial）、

## 数据结构

### 字符串string

| 方法           | 作用                                       |
| -------------- | ------------------------------------------ |
| `lower()`      | 转为小写                                   |
| `upper()`      | 转为大写                                   |
| `capitalize()` | 首字母大写，其他小写                       |
| `title()`      | 每个单词首字母大写                         |
| `casefold()`   | 类似 `lower()`，但更彻底（适合国际化比较） |

### 列表list

#### 增加元素

```python
lst = [1, 2]
lst.append(3)          # [1, 2, 3]
lst.extend([4, 5])     # [1, 2, 3, 4, 5]
lst.insert(1, 99)      # 在指定位置插入元素，[1, 99, 2, 3, 4, 5]
```

#### 删除元素

```python
lst = [10, 20, 30, 20]
lst.pop()        # → 删除 30
lst.remove(20)   # → 删除第一个 20
del lst[0]       # → 删除 10
lst.clear()      # → []
```

#### 查找与统计

| 操作     | 语法                  | 说明                         | 示例                  |
| -------- | --------------------- | ---------------------------- | --------------------- |
| 索引访问 | `lst[i]`              | 获取第 i 个元素（从 0 开始） | `nums[0] → 1`         |
| 负索引   | `lst[-1]`             | 获取最后一个元素             | `nums[-1] → 5`        |
| 切片     | `lst[start:end:step]` | 取子序列                     | `nums[1:4] → [2,3,4]` |
| 修改     | `lst[i] = x`          | 修改第 i 个元素              | `nums[0] = 10`        |
| 长度     | `len(lst)`            | 获取列表长度                 | `len(nums)`           |
| 是否存在 | `x in lst`            | 判断元素是否在列表中         | `3 in nums → True`    |

```python
lst = [1, 2, 3, 2, 2]
lst.index(3)  # 2，某元素第一次在list中出现的位置
lst.count(2)  # 3，某元素的个数
```

#### 排序与反转

```python
lst = [3, 1, 4, 2]
lst.sort()                     # [1, 2, 3, 4]
lst.sort(key = None, reverse = True)         # [4, 3, 2, 1]
lst.reverse()                  # [1, 2, 3, 4] → [4, 3, 2, 1]
words = ["apple", "banana", "pear"]
words.sort(key=len)  # ['pear', 'apple', 'banana']
```

#### 复制与组合

```python
a = [1, 2]
b = a.copy()
c = a + [3, 4]     # [1, 2, 3, 4]
d = a * 3          # [1, 2, 1, 2, 1, 2]
```

#### 列表推导式

```python
# 基本形式
[new_element for element in iterable if condition]
# 示例：
squares = [x**2 for x in range(5)]           # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0] # [0, 2, 4, 6, 8]
```

### 元组tuple

#### 创建

```python
# 常规写法
t = (1, 2, 3)

# 括号可以省略
t = 1, 2, 3

# 只有一个元素时，必须加逗号！
t1 = (1,)      # ✅ 正确：单元素元组
t2 = (1)       # ❌ 不是元组，是整数

# 空元组
empty = ()
```

#### 访问与操作

支持索引、切片访问：

```python
t = (10, 20, 30, 40)

print(t[0])      # 10
print(t[-1])     # 40
print(t[1:3])    # (20, 30)
```

| 操作     | 示例          | 说明         |
| -------- | ------------- | ------------ |
| 长度     | `len(t)`      | 返回元素个数 |
| 拼接     | `t1 + t2`     | 合并为新元组 |
| 重复     | `t * 3`       | 重复内容     |
| 成员判断 | `x in t`      | 判断是否包含 |
| 迭代     | `for x in t:` | 遍历元素     |
| 索引查找 | `t.index(x)`  | 返回元素索引 |
| 计数     | `t.count(x)`  | 元素出现次数 |

### 集合set

#### 创建

```python
# 用大括号 {}
s = {1, 2, 3, 4}

# 用 set() 构造函数（推荐）
s2 = set([1, 2, 2, 3])
print(s2)   # {1, 2, 3}  ← 自动去重

# 空集合必须用 set()
empty = set()
print(type(empty))  # <class 'set'>

# ❌ {} 默认是字典，不是集合
wrong = {}
print(type(wrong))  # <class 'dict'>
```

#### 基本操作

不支持索引访问，但支持for遍历！

| 操作     | 示例           | 说明                         |
| -------- | -------------- | ---------------------------- |
| 添加     | `s.add(x)`     | 添加单个元素                 |
| 删除     | `s.remove(x)`  | 删除指定元素（不存在会报错） |
| 安全删除 | `s.discard(x)` | 删除元素（不存在不报错）     |
| 弹出     | `s.pop()`      | 随机删除并返回一个元素       |
| 清空     | `s.clear()`    | 删除所有元素                 |
| 长度     | `len(s)`       | 元素个数                     |
| 成员判断 | `x in s`       | 判断是否存在                 |

#### 集合运算

```python
a = {1, 2, 3}
b = {3, 4, 5}
print(a & b)  # 交，{3}
print(a | b)  # 并，{1, 2, 3, 4, 5}
print(a - b)  # 差，{1, 2}
print(a ^ b)  # 对称差，{1, 2, 4, 5}
```

关于子集：

```python
a = {1, 2}
b = {1, 2, 3}

print(a < b)   # True  真子集
print(a <= b)  # True  子集（包含相等）
print(b > a)   # True  真超集
print(b >= a)  # True  超集
```

### 字典dict

#### 创建

```python
person = {"name": "Alice", "age": 25, "job": "Engineer"}
person = dict(name="Alice", age=25, job="Engineer")
pairs = [("name", "Alice"), ("age", 25)]
person = dict(pairs)
empty = {}
```

#### 访问与修改

| 操作           | 语法                 | 示例                         |
| -------------- | -------------------- | ---------------------------- |
| 访问值         | `d[key]`             | `person["name"] → 'Alice'`   |
| 修改值         | `d[key] = new_value` | `person["age"] = 26`         |
| 新增键值       | `d[new_key] = value` | `person["city"] = "Beijing"` |
| 删除键值       | `del d[key]`         | `del person["job"]`          |
| 判断键是否存在 | `key in d`           | `'age' in person → True`     |

#### 遍历

遍历键：

```python
for k in person.keys():
    print(k)
```

遍历值：

```python
for v in person.values():
    print(v)
```

遍历键值对：

```python
for k, v in person.items():
    print(k, "→", v)
```

#### 字典推导式

```python
# 基础
squares = {x: x**2 for x in range(5)}
print(squares)  # {0:0, 1:1, 2:4, 3:9, 4:16}

# 筛选
even_squares = {x: x**2 for x in range(6) if x % 2 == 0}
print(even_squares)  # {0:0, 2:4, 4:16}
```

## 语法/杂项

### 类型判断isinstance()

```python
# 字符串
isinstance("abc", str)           # ✅ True

# 数值类型
isinstance(123, int)             # ✅ True
isinstance(3.14, float)          # ✅ True
isinstance(2+3j, complex)        # ✅ True

# 布尔类型
isinstance(True, bool)           # ✅ True

# 列表 / 元组 / 字典 / 集合
isinstance([1, 2, 3], list)      # ✅ True
isinstance((1, 2, 3), tuple)     # ✅ True
isinstance({"a": 1}, dict)       # ✅ True
isinstance({1, 2, 3}, set)       # ✅ True

# 多类型判断（用元组）
x = 3.14
isinstance(x, (int, float))      # ✅ True：只要是其中一个类型就返回 True

# 是否是可迭代对象
isinstance([], Iterable)

# 是否是迭代器
isinstance([], Iterator)
```

### 切片

前两个表示边界、最后一个表示步长。

### 迭代

range在for循环中的应用：

```python
# 起点，终点，步长
for i in range(start, stop, step):
    ...
```

enumerate在for循环中的应用：

```python
# 返回的是idx索引（从0开始）和对应的元素
for idx , element in enumerate(data):
    ...
```

zip在for循环中的应用：

```python
list1 = []
list2 = []
# 同步迭代多个列表
# 如果列表长度不同，zip() 只会匹配最短长度，多余的元素会被丢弃
for element1, element2 in zip(list1, list 2):
    ...
```

enumerate和zip在for循环中结合使用：

```python
for step, (stream,replay) in enumerate(zip(stream_train_dataloader,replay_train_dataloader)):
```

### 列表推导式

```
a = [x for ... if ...]
```

### 生成器 generator

是一种可迭代对象，返回`Iterator`对象表示的是一次性数据流，能迭代但不直接显示内容。

相比列表生成式，一边计算，一边逐个产出数据，而且只能用一次，节省内存。

#### 用 yield 定义生成器函数

调用它不会立即执行，而是返回一个生成器对象。

执行过程：每次循环调用 `next(gen)`，函数从上次的 `yield` 处继续执行。

```python
def func(n):
    a = [1]
    for _ in range(n):
        yield a
        a = [1] + [a[i] + a[i + 1] for i in range(len(a) - 1)] +[1]

if __name__ == '__main__':
    fn = func(6)
    for d in fn:
        print(d)
```

#### 用生成器表达式（类似列表推导式）

```python
gen = (x**2 for x in range(5))
print(gen)  # <generator object <genexpr> at ...>

for val in gen:
    print(val)
```

## 函数式编程

函数式编程的一个特点就是，允许把函数本身作为参数传入另一个函数，还允许返回一个函数。

### map/reduce

map()函数用于序列$\to$序列。接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。由于Iterator是惰性序列，因此通过list()函数让它把整个序列都计算出来并返回一个list。

```python
def pow(x):
    return x ** 2

if __name__ == '__main__':
    a = map(pow, [1, 2, 3, 4, 5])
    print(list(a))
```

reduce()用于序列$\to$obj。把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，其效果就是：

```python
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```

比如把str(13579)转换成int(13579)：

```python
from functools import reduce
def fn(x, y):
    return x * 10 + y
def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]
reduce(fn, map(char2num, '13579'))
13579
```

### filter

用于过滤序列。和map()类似，filter()也接收一个函数和一个序列。和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。

由于返回的是Iterator对象，同样需要list转换。

```python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```

### sorted

完整的参数有三个：list，key函数，reverse: bool

直接对list进行排序：

```python
>>> sorted([36, 5, -12, 9, -21])
[-21, -12, 5, 9, 36]
```

sorted()函数也是一个高阶函数，可以接收一个key函数来实现自定义的排序：

原理是key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序。

```python
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

def by_name(t):
    return t[0].lower()

L2 = sorted(L, key=by_name)
print(L2)
```

要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True：

```python
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
['Zoo', 'Credit', 'bob', 'about']
```

### 匿名函数 lambda

用于实现简单函数，只能有一个表达式，常和之前说的map reduce fiter sort连用。

```python
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

L2 = sorted(L, key = lambda t: t[0].lower())
print(L2)
```

### 装饰器 decorator

在不改变原函数定义的要求下，为原函数增加额外功能的写法。

本质上，decorator 就是一个返回函数的高阶函数。

**在函数定义阶段，装饰器函数本身执行一次；之后每次调用被装饰的函数时，执行的都是装饰器返回的内部包裹函数（wrapper）。**

```python
def decorator(func):
    print("装饰器执行了")
    def wrapper(*args, **kw):
        print(f"wrapper执行：{func.__name__}")
        func()
    return wrapper

@decorator
def say_hi():
    print("Hi!")

# say_hi()
```

@decorator 等价于 say_hi = decorator(say_hi)。

* 当函数执行到@decorator，会立即调用一次 decorator(say_hi) 把返回值 wrapper 赋给 say_hi。
* 之后每次执行 say_hi()，其实是在执行 wrapper()。

### 偏函数 partial

固定一个函数的部分参数，生成一个新的函数，让调用更简单。

简单例子：

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# 创建一个“平方函数”
square = partial(power, exponent=2)

print(square(3))  # 3 ** 2 = 9
print(square(5))  # 5 ** 2 = 25
print(square(8, exponent = 3)) #这里必须是命名关键字参数的形式
```

与高阶函数结合：比如map和filter，用偏函数解决只能传一个参数

```python
from functools import partial

def multiply(x, y):
    return x * y

double = partial(multiply, y=2)

nums = [1, 2, 3, 4]
result = list(map(double, nums))
print(result)
```

## 模块

一个.py文件就是一个模块。

包：按目录组织模块。

### 作用域

* 公开变量/函数：可以被直接引用。
* 特殊变量/函数：`__xxx__`，可以被直接引用，但是有特殊用途，我们自己的变量一般不要用这种变量名。
* 私有变量/函数：`__xxx`，不应该被直接引用。

### 模块搜索路径

当我们试图加载一个模块时，Python会在指定的路径下搜索对应的.py文件，如果找不到，就会报错。

## 类和对象

### 基础

类的定义：

```python
class Student(object):
    pass
```

创建实例：

```python
bart = Student()
```

绑定属性：

```python
bart.name = 'Bart Simpson'
```

公开变量：支持自由修改属性，`bart.name = 'Bart Simpson'`。

私有变量：在属性名称前加两个下划线`__`，只有内部可以访问，外部不能访问。但是可以通过调用函数，实现访问。

继承：

* 子类继承父类的全部功能。
* 子类也可以增加一些方法。
* 覆盖：子类对父类方法进行修改，运行时调用的是修改后的方法。
* 多态：子类的数据类型既是自己，又是父类。在调用以父类为参数的函数时，所有子类对象都能适用。

使用dir()：获得一个对象的所有属性和方法。

类属性：公共属性，归类所有，但类的所有实例都可以访问到。实例修改，仅对该实例生效，类属性不会消失。

### @property

把调用方法简化为调用属性。

### init方法

`super().__init__()`：重写父类构造函数，init()里面填父类构造函数的参数

```python
class Parent:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Child(Parent):
    def __init__(self, a, b, c):
        super().__init__(a, b)  # 把 a、b 传给父类
        self.c = c              # 子类自己的属性
```

### __call__方法

`__call__`：实现让对象能像函数一样被调用

```python
class MyClass:
    def __call__(self, x):
        print(f"你传入了 {x}")

obj = MyClass()

obj(123)   # 等价于 obj.__call__(123)
```

### len方法

`__len__`：获取一个对象的长度

```python
class MyDog(object):
	def __len__(self):
		return 100

dog = MyDog()
len(dog)
```

### str方法

`__str__`：一个返回值**必须**为 `str` 类型的方法，规定了**实例**转化为 `str` 类型的值。

```python
class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Student object (name: %s)' % self.name
print(Student('Michael'))
Student object (name: Michael)
```

### iter方法

如果一个类想被用于`for ... in`循环，类似list或tuple那样，就必须实现一个`__iter__()`方法，该方法返回一个迭代对象，然后，Python的for循环就会不断调用该迭代对象的`__next__()`方法拿到循环的下一个值，直到遇到`StopIteration`错误时退出循环。

```python
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b

    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值

>>> for n in Fib():
...     print(n)
...
1
1
2
3
5
...
46368
75025
```

### getitem方法

`__getitem__`：既保证迭代功能，又能实现像list那样下标索引、切片。

```python
class Fib(object):
    def __getitem__(self, n):
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a
```

### getattr方法

`__getattr__`：用于处理调用的类的方法和属性不存在的情况，可以动态地返回属性和方法





