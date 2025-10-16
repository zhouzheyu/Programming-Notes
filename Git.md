# Git

Git是目前世界上最先进的分布式版本控制系统。

* 回溯修改历史
* 多人协作编辑
* 工作区-暂存区-版本库

本文参考[廖雪峰老师的Git教程](https://liaoxuefeng.com/books/git/introduction/index.html)。

## 写在前面

你说的对，git牛逼。

## 安装Git

安装好之后配置用户名和邮件：

```bash
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```

 ## 创建版本库

先cd到想要git的目录：

```bash
$ mkdir learngit
$ cd learngit
$ pwd
/Users/michael/learngit
```

### git init

把当前目录变成Git可以管理的仓库：

当前目录下多了一个.git的目录，这个目录是Git来跟踪管理版本库的，不用管。

如果没看到.git目录，那是因为这个目录默认隐藏，用`ls -ah`命令就可以看见。

```bash
$ git init
Initialized empty Git repository in /Users/michael/learngit/.git/
```

### 把文件添加到版本库

#### git add file

把文件添加到暂存区：

```bash
$ git add readme.txt
```

#### git commit -m "string"

把暂存区中的文件提交到当前分值

-m后的是修改声明。

返回的结果：多少文件被改动、插入多少行、删除多少行

```bash
$ git commit -m "wrote a readme file"
[master (root-commit) eaadf4e] wrote a readme file
 1 file changed, 2 insertions(+)
 create mode 100644 readme.txt
```

其他可能用的到git add命令：

| 命令             | 作用                                                       |
| ---------------- | ---------------------------------------------------------- |
| `git add <file>` | 把单个文件加入暂存区                                       |
| `git add <dir>/` | 把整个目录加入暂存区                                       |
| `git add .`      | 添加当前目录下的**所有新增和修改的文件**（不一定包括删除） |
| `git add -A`     | 添加**所有更改（新增、修改、删除）** ✅ 推荐                |
| `git add -u`     | 只添加**已被跟踪文件**的修改或删除（不包括新建文件）       |

## 时光机穿梭

### git status

状态查询命令：哪些文件改了、哪些还没提交、当前在哪个分支上。

修改readme.txt内容，未提交，显示结果：

```bash
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

### git diff

`git diff filename`：如果当前文件没提交，查询和仓库中不同的地方。

也可以直接写`git diff`，对比仓库的区别。

```bash
$ git diff readme.txt 
diff --git a/readme.txt b/readme.txt
index 46d49bf..9247db6 100644
--- a/readme.txt
+++ b/readme.txt
@@ -1,2 +1,2 @@
-Git is a version control system.
+Git is a distributed version control system.
 Git is free software.
```

### 版本回退

版本1：wrote a readme file

版本2：add distributed

版本3：append GPL

#### git log

查看历史记录：包括commit id, Author, Date

```bash
$ git log
commit 1094adb7b9b3807259d8cb349e7df1d4d6477073 (HEAD -> master)
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 21:06:15 2018 +0800

    append GPL

commit e475afc93c209a690c39c13a46716e8fa000c366
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 21:03:36 2018 +0800

    add distributed

commit eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 20:59:18 2018 +0800

    wrote a readme file
```

`git log --pretty=oneline`：简化查看

```bash
$ git log --pretty=oneline
1094adb7b9b3807259d8cb349e7df1d4d6477073 (HEAD -> master) append GPL
e475afc93c209a690c39c13a46716e8fa000c366 add distributed
eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0 wrote a readme file
```

#### git reset --hard xxx

Git在内部有个指向当前版本的`HEAD`指针，当你回退版本的时候，Git仅仅修改HEAD指向。

`git reset --hard HEAD~x`：回退到前x个版本。

或者写成：`git reset --hard HEAD^^`，表示回退到前2个版本

```bash
$ git reset --hard HEAD^
HEAD is now at e475afc add distributed
```

此时使用`git log`，就只能看到当前版本及以前的记录。

想要找回修改前的版本，用commit id：只写前面一部分就可以

```bash
$ git reset --hard 1094a
HEAD is now at 83b0afe append GPL
```

#### git reflog

记录每一条历史命令，最前面的是commit id

```bash
$ git reflog
e475afc HEAD@{1}: reset: moving to HEAD^
1094adb (HEAD -> master) HEAD@{2}: commit: append GPL
e475afc HEAD@{3}: commit: add distributed
eaadf4e HEAD@{4}: commit (initial): wrote a readme file
```

### 工作区和暂存区

工作区：在电脑里能看到的目录

版本库：是工作区中的隐藏目录.git，包含以下内容

* 称为stage（或者叫index）的暂存区
* Git为我们自动创建的第一个分支master
* 指向master的HEAD指针

工作原理：git add把文件提交到暂存区，然后通过git commit指令，把暂存区中的所有修改一次性全部交到master分支。

### 管理修改

Git跟踪并管理的是修改，而非文件。

如果使用以下操作流程：

第一次修改 -> git add -> 第二次修改 -> git commit

那么因为 git commit 交的是暂存区中的修改，所以与第二次修改无关。

### 撤销修改

#### git restore -- file

丢弃工作区的修改。

本质上是用版本库里的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。

一种是自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；

一种是已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。

`git restore -- file`命令中的`--`很重要。`--`是 Git 命令解析里“**分隔符**”，它告诉Git，从这里开始，后面的是文件名，不是分支名或参数。

### 删除文件

#### rm file

在工作区删除文件。

#### git rm file

在版本库中删除该文件。

#### 误删

用git restore -- file还原。

## 远程仓库

### 创建SSH Key

输入：`ls -al ~/.ssh`

如果看到类似文件：id_ed25519    id_ed25519.pub，说明已经有 SSH 密钥，可以直接使用。

首次配置，需要生成 SSH Key（ed25519）：`ssh-keygen -t ed25519 -C "your_email@example.com"`。

一般对这个Key无需设置密码。

复制公钥：`pbcopy < ~/.ssh/id_ed25519.pub`

登陆GitHub，添加SSH Key，粘贴公钥。

为什么GitHub需要SSH Key呢？因为GitHub需要识别出你推送的提交确实是你推送的，而不是别人冒充的，而Git支持SSH协议，所以，GitHub只要知道了你的公钥，就可以确认只有你自己才能推送。

当然，GitHub允许你添加多个Key。假定你有若干电脑，你一会儿在公司提交，一会儿在家里提交，只要把每台电脑的Key都添加到GitHub，就可以在每台电脑上往GitHub推送了。

### 添加远程库

已经在本地创建了一个Git仓库后，又想在GitHub创建一个Git仓库，并且让这两个仓库进行远程同步，这样，GitHub上的仓库既可以作为备份，又可以让其他人通过该仓库来协作。

在本地的仓库下运行命令：

```bash
git remote add origin git@github.com:username/projectname.git 
```

添加后，远程库的名字就是origin，这是Git默认的叫法，也可以改成别的，但是origin这个名字一看就知道是远程库。

git remote add origin git@github.com:zhouzheyu/Commands.git

把本地库的所有内容推送到远程库上：

```bash
git push -u origin main
```

-u：首次推出main分支，不仅把本地的main分支内容推送的远程新的main分支，还会把本地的main分支和远程的main分支关联起来，在以后的推送或者拉取时就可以简化命令。

从现在起，只要本地作了提交，就可以通过命令：

```bash
git push origin main
```

查看远程库信息`git remote -v`：

```bash
$ git remote -v
origin  git@github.com:username/projectname.git (fetch)
origin  git@github.com:username/projectname.git (push)
```

删除远程库`git remote rm <name>`：

```bash
$ git remote rm origin
```

