# AlphaGen

Automatic formulaic alpha generation with reinforcement learning.

> Original work: *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning*, KDD 2023.  
> Repository: [github.com/ICT-FinD-Lab/alphagen](https://github.com/ICT-FinD-Lab/alphagen)  
> Maintained by MLDM research group, [IIP, ICT, CAS](http://iip.ict.ac.cn/).

## 快速使用

### 1. 数据准备

将原始 CSV 数据转换为 qlib 二进制格式：

```bash
# 1) 确保 qlib_data/ 目录下有 ins_data_all.csv（多品种单文件）
# 2) 按品种拆分 CSV 并对齐到临时目录
# 3) 运行 dump 脚本
uv run python -m data_collection.qlib_dump_bin dump_all \
    --csv_path qlib_data/tmp \
    --qlib_dir qlib_data \
    --freq 5min \
    --date_field_name datetime \
    --exclude_fields "index,code,time"
```

输出结构：
```
qlib_data/
├── calendars/
│   └── 5min.txt      # 交易日历（5 分钟线）
├── instruments/
│   └── all.txt       # 品种列表及起止时间
└── features/
    └── {code}/        # 每个品种一个子目录
        ├── open.5min.bin
        ├── high.5min.bin
        ├── low.5min.bin
        ├── close.5min.bin
        ├── volume.5min.bin
        ├── amount.5min.bin
        └── ...
```

### 2. 配置文件

所有运行参数集中在 `symbol_config.json`，主要节点：

| 节点 | 说明 | 关键字段 |
|------|------|---------|
| `qlib_data_path` | 数据目录 | `"qlib_data"` |
| `instruments` | 品种集合 | `"all"`（全部）或单品种 |
| `device` | 计算设备 | `"cpu"` / `"cuda:0"` |
| `freq` | 数据频率 | `"5min"` |
| `data` | 训练/测试时间段 | `train_start`, `train_end`, `test_segments` |
| `target_horizon` | 预测目标步数 | `20`（5min 线 20 步 ≈ 100 分钟） |
| `llm` | LLM API 配置 | `base_url`, `api_key`, `model`, `model_max_tokens` |
| `rl` | 强化学习超参 | `pool_capacity`, `ppo`, `lstm_network`, `steps_default` |
| `llm_only` | 纯 LLM 实验参数 | `pool_size`, `n_replace`, `n_updates` |
| `backtest` | 回测参数 | `benchmark`, `top_k`, 手续费等 |
| `gp` / `dso` | 基线方法参数 | GP 进化参数 / DSO 训练参数 |
| `paths` | 输出路径 | `save`, `tensorboard`, 各测试输出目录 |

临时测试建议创建独立配置文件（如 `test_config.json`），使用极小时间段验证流程。

### 3. 运行脚本

从项目根目录以模块方式运行：

```bash
# 主实验 / 强化学习 alpha 挖掘
uv run python -m scripts.rl \
    --config_path test_config.json \
    --steps 50 --pool_capacity 2

# 纯 LLM 迭代生成 alpha
uv run python -m scripts.llm_only \
    --config_path symbol_config.json \
    --pool_size 5 --n_updates 3

# LLM 输出有效性测试
uv run python -m scripts.llm_test_validity \
    --config_path symbol_config.json \
    --n_repeats 5

# 遗传规划基线
uv run python gp.py 0 --config_path test_config.json

# 深度符号回归基线
uv run python dso.py 0
```

> **注意**：测试时务必使用极短时间范围（如 `test_config.json`），避免全量运行导致长时间计算。

### 4. 输出

- Model checkpoint & alpha pool → `paths.save` 目录
- TensorBoard 日志 → `paths.tensorboard` 目录
- 回测结果 → `paths.backtest_output` 目录

---

## Citing our work

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```

## Contributors

This work is maintained by the MLDM research group, [IIP, ICT, CAS](http://iip.ict.ac.cn/).

Maintainers include:

- [Hongyan Xue](https://github.com/xuehongyanL)
- [Shuo Yu](https://github.com/Chlorie)

Thanks to the following contributors:

- [@yigaza](https://github.com/yigaza)
