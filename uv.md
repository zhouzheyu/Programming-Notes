# uv

下载uv包：

```
pip install uv
```

创建虚拟环境：

```
uv venv --python 3.11
```

激活虚拟环境：

```
source .venv/bin/activate
```

退出虚拟环境：

```
deactivate
```

同步依赖：

```
uv sync
```

运行文件（同时根据uv lock中的内容同步依赖）：

```
uv run main.py
```

在虚拟环境中安装依赖：

```
uv pip install requests
```

```
uv add requests
```

在虚拟环境中删除依赖：

```
uv remove requests
```

查看当前虚拟环境中的所有包：

```
uv pip list
```

查看项目依赖树：

```
uv tree
```

