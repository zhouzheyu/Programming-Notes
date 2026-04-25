# cursor

用户主页：https://cursor.com/cn/dashboard

文档：https://cursor.com/cn/docs

命令面板：cmd + shift + P

### Agent

command + I

**Agengt Tips:**

* plan模式：负责理解需求、拆解任务、给出实现思路/架构设计；不会自动修改代码/执行命令
* agent模式：改代码和执行命令，注意@的使用。
* slash指令：快捷指令。比如/pr...
* 可以给agent传图片
* Fork Chat：复制当前对话，开一个“分支版本”继续聊
* roll back: discard changes



## 核心概念

### Tab

* 修改/补全多行代码
  * 添加文本时，补全会以半透明的灰显文本出现。
  * 修改现有代码时，会在当前行右侧显示差异弹窗。
  * 按 `Tab` 接受建议，按 `Escape` 拒绝，或用 `Cmd + Arrow Right` 逐词接受。继续输入或按 `Escape` 可隐藏建议。
* 在文件内及跨文件跳转
  * 文件内跳转：Tab会预测下一个编辑位置。接受编辑后，再按 `Tab` 即可跳转。
  * 跨文件跳转：Tab 会在多个文件间预测上下文感知的编辑。当建议跨文件跳转时，底部会出现一个传送窗口。
* 自动导入
  * 在缺少时自动添加 import 语句。

修改tab可以在哪些文件中使用：右下角状态栏。

### 内联编辑

* 编辑选区
  * 先选择代码，然后按 command + K，打开内联编辑器。
  * 在未选中任何内容时，Cursor 会在光标位置生成新代码。
* 快速提问
  * 在内联编辑器中按下 Opt + Return。
* 整文件编辑
  * 按command + L进入右侧chat，输入指令，command + shift + return。
  * 其实就是command + I，废话...
* 发送到chat
  * 选择代码，command + L将选区发送到 Chat。
  * 可以单文件，也可以拓展到多文件编辑。
* 后续指令
  * 每次编辑后，添加补充说明并按下 Return 以优化结果。AI 会根据你的反馈更新结果。



