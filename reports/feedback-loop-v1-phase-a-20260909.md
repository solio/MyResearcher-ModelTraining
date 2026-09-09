# Feedback Loop V1 Phase A — DataClean → Feed → Prediction → Human Feed

状态：`PHASE_A_DISPATCHED_PENDING_HUMAN_LABELS`

本阶段只完成真实输入的准备、冻结快照和手机标注任务派发；没有训练、调参或改变任何模型/artifact。

## 来源与身份

- 输入：DataClean round-003 `pilot-300.jsonl`，300 条，SHA-256
  `6c3cef7851626d83eb86296799b63273eddfd2bee15891f1ecbe1c6c7e6db3de`。
- stock scope：collector scope `stock:601012`；`stock_name=隆基绿能` 来自
  DataCollector `config/targets.short-term.json` 的只读 `stock_names` 映射。
- 当前七头候选仍为 `m2-selected-candidate-frozen-for-m3-20260904`，content address
  `8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf`。
- M4 Human100 treatment 已关闭，结论为 `HUMAN_SUPERVISION_NO_GAIN_DIAGNOSTIC`；
  既有候选未改变。当前仓库只有本地 `main` 和一个 primary worktree。

## 固定 14 条身份与划分

`sample_id=fl1-<observation_id>`；14 条来自不同 `exact_content_group`。选择规则在人工标注前冻结为 7 条模型不确定/边界 + 7 条长度与 `sampling_reasons` 覆盖；每组各 5 条进入 Human Train、各 2 条进入 Human Check。

| role | sample_id | observation_id | exact_content_group | stratum |
|---|---|---|---|---|
| Train | `fl1-34ee949ab855b6a92f3648fb9f4567207e6292ca72134b5d33ed4df01b7ecf2d` | `34ee949ab855b6a92f3648fb9f4567207e6292ca72134b5d33ed4df01b7ecf2d` | `4bfeb88d40be1d359c14a1efa4c3e0f32d282815d2278c09f5c0f2b0c3c54e2a` | boundary |
| Train | `fl1-6cb34337977b8e80553eb6f99fba7f96efa9778424805cb2bdf0b0b80235e8d1` | `6cb34337977b8e80553eb6f99fba7f96efa9778424805cb2bdf0b0b80235e8d1` | `0e4db56d7ee81b6d1ddcf15474b4f241332d945e634326d3f9c82ff7b97c602d` | boundary |
| Train | `fl1-53259e172b41b99acaaaa47f2acf18f86d70b51713f7081f33f83d8e334ea4a6` | `53259e172b41b99acaaaa47f2acf18f86d70b51713f7081f33f83d8e334ea4a6` | `c776d2992a29cb567ea0f78cf3b6ab6f7e62bc9eb06fe424dbc65f25aa196d94` | boundary |
| Train | `fl1-257df71258e0f3d3592d5e5e5a656a9f252c7d2479cbc4222f928509e8405242` | `257df71258e0f3d3592d5e5e5a656a9f252c7d2479cbc4222f928509e8405242` | `2fcfa0ee6ced36322c9dd238f6d8bdecbbf0851e579956a190c9338b06746556` | boundary |
| Train | `fl1-24f3bb6ff725be2ab18ae078430b76836700da21a57bc6ededbb3c14876c9187` | `24f3bb6ff725be2ab18ae078430b76836700da21a57bc6ededbb3c14876c9187` | `a2f37d8c6106ff337f90c9bb35889ffeaebe23b9eec24904c6a098de471f4d1d` | boundary |
| Train | `fl1-7533a1a1b87b36e7c3ae81196d4f95c90294ed349acc9fd0462cd2f9b9c10b1b` | `7533a1a1b87b36e7c3ae81196d4f95c90294ed349acc9fd0462cd2f9b9c10b1b` | `a081965e34047c9f67eb6dfea3070efe3f159a035b4dca695352a98aca31f59a` | coverage |
| Train | `fl1-3f8d639a4cfce44cbe5dc8bbc53cb0d256ae0ced1e5ef1e6275e111a3625a636` | `3f8d639a4cfce44cbe5dc8bbc53cb0d256ae0ced1e5ef1e6275e111a3625a636` | `ba3f4cbe10239db5b3e79daf9f3b9be9525d36a10b97be773da4f1240b60a9d1` | coverage |
| Train | `fl1-43e3eb044f431a4d8e07acc815742f78aaf3c358ccd7f2c7ed33b5e7e2f8488c` | `43e3eb044f431a4d8e07acc815742f78aaf3c358ccd7f2c7ed33b5e7e2f8488c` | `24ad73ef966b6c5e919f2c737a2464b02afbfd56ac238c4682d549a33113d2ee` | coverage |
| Train | `fl1-3b79ad7e81beaaa17d5e3a67d8eed1a4c890397809c3de59f0dc916781b41689` | `3b79ad7e81beaaa17d5e3a67d8eed1a4c890397809c3de59f0dc916781b41689` | `51e0a596449f9a5587f0652fdf9e2779fd5e9dcc367b785e5c48cc5748802c39` | coverage |
| Train | `fl1-c4c78ca17a06e66bbf0031df70c0378a67a7ad9ac592abc3e9c85acf1d4b1bb8` | `c4c78ca17a06e66bbf0031df70c0378a67a7ad9ac592abc3e9c85acf1d4b1bb8` | `3adda5c742e053680334e7f213d978f7e102642f5cf4c4877a2e1db8ca7109a4` | coverage |
| Check | `fl1-ae2c89b7c6f65502531e987be111df0ef8223e0f40865af4f71135bf103c4451` | `ae2c89b7c6f65502531e987be111df0ef8223e0f40865af4f71135bf103c4451` | `84c2b58842f40bd605118589c149c03972baa201d3ce8b460e75ef92c902fe93` | boundary |
| Check | `fl1-5202be71c30b2bd311652af7dc6146873db860612e19571e758b97c03dbae3f5` | `5202be71c30b2bd311652af7dc6146873db860612e19571e758b97c03dbae3f5` | `cf277ddab2a5c2f67d43e51788d6b4cc57a7f34d71da41fc5591bc707bd87a5f` | boundary |
| Check | `fl1-ed18ed70b6dc52f1004bc9883fb58033b1a13e45fef9be6029db5e661b263b7f` | `ed18ed70b6dc52f1004bc9883fb58033b1a13e45fef9be6029db5e661b263b7f` | `ebc701a2261aa5d5d20887c6c7b1a8971f7a62c81ba242ab362db828c941277d` | coverage |
| Check | `fl1-3c188dd343f38b70471a7b227e0e93d2406098c7b78948dd6b2946fe5a0ca3a5` | `3c188dd343f38b70471a7b227e0e93d2406098c7b78948dd6b2946fe5a0ca3a5` | `138cde15cb0a12bb1319cc67699a1f997f7c72d35202eed60b315155ec316efe` | coverage |

## 标注入口与状态

- 手机入口：现有 `MyResearcher-Labeling-app`，批次 `fl1-train-20260909` 为 10 samples × 7 heads = 70 tasks；`fl1-check-20260909` 为 4 samples × 7 heads = 28 tasks。
- 任务载荷只含 `sample_id`、文本和安全元数据，不含模型预测/概率；预测前快照保存在
  `runs/feedback-loop-v1-20260909/prediction-snapshot.jsonl`（300 条），供后续诊断而非标注预填。
- `status` 已实际确认：Train `total=70, done=0, finals=0`；Check `total=28, done=0, finals=0`。
- 准备清单及文件 hash：`runs/feedback-loop-v1-20260909/prepare-manifest.json`，content address
  `25ad27f64b6cc3e0348655c8f9a8717144e0812dd381a1ef34fa1cadab4c8c78`。

用户完成手机标注后唯一 pull 命令（不使用 `--final-only`，保留可修订 disposition/时间/批次/schema）：

```bash
PYTHONPATH=src .encoder-venv/bin/python tools/feedback_loop_v1.py pull \
  --batch fl1-train-20260909 --batch fl1-check-20260909 \
  --output-jsonl runs/feedback-loop-v1-20260909/human-labels-revisable.jsonl
```

Pull 输出按 `sample_id/head` 合并最新 final answer，并保留 `disposition`、`updated_at`、`batch` 和 `schema_version`；它是可修订 Human Feed 快照，不是永久 Gold，也不会自动进入训练。

本阶段明确未访问 Test、Anchor、Gold、OOD、LLM 或生产数据；没有训练或修改模型。M4 负结果不会决定新人工标签的方向，Phase B/C 不在本轮范围内。
