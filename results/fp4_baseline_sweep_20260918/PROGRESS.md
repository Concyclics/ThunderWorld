# 吞吐扫描进度

已完成 20/20 个方法×输入组合。扫描 complete=True。

这是 PRO 6000 固定实现的诊断数据；B>1 按用户要求只测速度，未验证并发质量。B 表示独立完整请求并发，不是合并 GEMM batch。

| 方法 | 输入 | B1 视频/s | B2 | B4 | B8 | 状态 |
|---|---|---:|---:|---:|---:|---|
| bf16 | rbench | 0.5805 | 0.7283 | 0.7417 | 0.6767 | observed_two_doubling_plateau_below_5_percent |
| bf16 | paibench | 0.5729 | 0.7174 | 0.7291 | 0.6707 | observed_two_doubling_plateau_below_5_percent |
| fp8 | rbench | 0.6587 | 0.8542 | 0.8857 | 0.7872 | observed_two_doubling_plateau_below_5_percent |
| fp8 | paibench | 0.6498 | 0.8363 | 0.8636 | 0.7737 | observed_two_doubling_plateau_below_5_percent |
| fp4_base | rbench | 0.6488 | 0.8974 | 0.9736 | 0.8695 | maximum_predeclared_batch_reached |
| fp4_base | paibench | 0.6367 | 0.8764 | 0.9584 | 0.8724 | maximum_predeclared_batch_reached |
| b1.5_protected_n | rbench | 0.5707 | 0.7770 | 0.8552 | 0.7918 | maximum_predeclared_batch_reached |
| b1.5_protected_n | paibench | 0.5580 | 0.7622 | 0.8345 | 0.7691 | maximum_predeclared_batch_reached |
| b2_joint | rbench | 0.5063 | 0.7194 | 0.8049 | 0.7429 | maximum_predeclared_batch_reached |
| b2_joint | paibench | 0.4954 | 0.7003 | 0.7817 | 0.7259 | maximum_predeclared_batch_reached |
| awq_marlin | rbench | 0.5618 | 0.6548 | 0.6616 | 0.5966 | observed_two_doubling_plateau_below_5_percent |
| awq_marlin | paibench | 0.5544 | 0.6462 | 0.6539 | 0.5876 | observed_two_doubling_plateau_below_5_percent |
| arcquant | rbench | 0.4215 | 0.3806 | 0.2583 | — | observed_two_doubling_plateau_below_5_percent |
| arcquant | paibench | 0.4161 | 0.3855 | 0.2848 | — | observed_two_doubling_plateau_below_5_percent |
| b2.5_ffn_guarded | rbench | 0.4564 | 0.6473 | 0.7480 | 0.6870 | maximum_predeclared_batch_reached |
| b2.5_ffn_guarded | paibench | 0.4473 | 0.6348 | 0.7261 | 0.6702 | maximum_predeclared_batch_reached |
| arcquant100 | rbench | 0.4004 | 0.3817 | 0.2883 | — | observed_two_doubling_plateau_below_5_percent |
| arcquant100 | paibench | 0.3969 | 0.3792 | 0.3021 | — | observed_two_doubling_plateau_below_5_percent |
| sharq_perf | rbench | 0.4191 | 0.3388 | 0.1859 | — | observed_two_doubling_plateau_below_5_percent |
| sharq_perf | paibench | 0.4161 | 0.3437 | 0.1992 | — | observed_two_doubling_plateau_below_5_percent |
