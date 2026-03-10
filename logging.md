# logging

一般会写一个logger工具函数：

```python
def get_logger(name: str = __name__) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

然后在模块里：

```python
logger = get_logger(__name__)

logger.info("Training EGTP head")
```

输出：

```bash
2026-03-10 12:00:00 | INFO | egtp.trainer | Training EGTP head
```

常见的日志等级：

```python
# 调试信息
logger.debug("feature dim = %d", x.shape[-1])
# 正常运行信息
logger.info("epoch %d finished", epoch)
# 警告信息
logger.warning("gradient exploding")
# 错误信息
logger.error("model load failed")
```

机器学习常见的：

```python
# 数据规模
logger.info(f"Train size: {len(train_x)}")
# 训练进度
logger.info("Training EGTP head")
logger.info(f"Epoch [{epoch}/{num_epochs}] loss={loss:.4f}")
...
logger.info("End training")
# 模型信息
logger.info(f"Model hidden size: {hidden_dim}")
# 评测结果
logger.info("Evaluating...")
logger.info(f"Test MAE={mae:.3f}")
```







