# tmux使用

让我的程序在服务器后台一直运行，即使断网、关电脑或退出终端也不会中断。

当前有哪些session：

```
tmux ls
```

开一个 session：

```bash
tmux new -s gen_len
```

运行程序：

```bash
python gen_output.py
```

本地退出，在服务器上继续跑（关键操作）：

```bash
Ctrl + B 然后按 D
```

本地回到当前最新进展：

```bash
tmux attach -t gen_len
```



```
python gen_output.py > log.txt 2>&1
```

