# GridWorld DQN Demo

FastAPI + WebSocket で動くシンプルな GridWorld 学習デモです。Ver3 では「過去の観測履歴」を Transformer で統合する **記憶付き DQN（CNN + Transformer）** に進化しました。ブラウザから学習の進み具合をリアルタイムに確認できます（ランダム迷路 + メモリ課題対応）。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # torch は CPU 版が入ります
# ※ OS/CPU によっては torch のインストールに数分かかります。失敗する場合は公式の CPU ホイール
#   (例: `pip install torch --index-url https://download.pytorch.org/whl/cpu`) で入れてください。
```

## 起動

```bash
uvicorn backend.app.main:app --reload
```

ブラウザで `http://localhost:8000/` を開くと UI が表示されます。

## アプリの使い方

- **Start**: 学習/実行を開始します。
- **Pause**: 一時停止します。
- **Reset**: Q テーブルと環境を初期化します。
- **Toggle AI**: 学習 ON/OFF を切り替えます（OFF でもランダム行動）。
- **Speed 1x/5x/20x**: 更新間隔を変更します。
- **Planner: DQN / MPC**: 行動選択を通常の DQN か、ワールドモデルを用いた MPC に切り替えます。
- WebSocket に `{"type": "maze_mode", "value": false}` を送ると固定迷路モードに戻ります（デフォルトはランダム迷路）。
- WebSocket に `{"type": "maze_mode", "value": "memory"}` を送ると「記憶が必要なループ/T 字路迷路」モードになります。
- WebSocket に `{"type": "planner_mode", "value": "mpc"}` / `{"type": "planner_mode", "value": "dqn"}` を送るとプランナーを切り替えられます。

ステータス欄で `episode`, `step`, `epsilon`, `total_reward`, 直近エピソードの成功率などがリアルタイムに更新され、Canvas 上でエージェントの動きが見られます。学習が進むほどゴールまでの経路が短くなる様子が確認できます。
Ver3 では追加で `last_loss`, `avg_recent_loss`, `avg_episode_reward` も表示されます。
Ver4 ではさらに `planner_mode`, `wm_last_loss`, `wm_avg_recent_loss` を表示しています。

## 仕様メモ

- GridWorld (10x10) に壁・スタート・ゴール・罠を配置。ランダム迷路は必ず「罠を踏まないスタート→ゴール経路」が存在するよう生成・検証しています。
- 観測はグリッド + エージェントを 8 で描画した「疑似画像」。学習入力はエージェント中心の 5x5 パッチ（壁外は 1 埋め）。
- 報酬: ゴール +1, ステップ -0.01, 壁衝突 -0.05, 罠 -1 (終了) に加え、最短距離が縮むとわずかに加点（デフォルト ON）。
- 1 エピソード最大 200 ステップ。エージェントは DQN (CNN + experience replay + target network) で、ε-greedy / ε 減衰。
- WebSocket で各ステップの状態を配信 (grid, agent_pos, episode/step, reward/total_reward, epsilon, success_rate, done など)。追加で `observation_patch`, `q_values`, `random_maze` なども送信します。

乱数シードはデフォルトで固定済みです。

## Ver3 (Transformer) のポイント

- 過去 **N=16 ステップ** の 5x5 観測パッチと「前回行動」「前回報酬」をトークン化し、CNN で埋め込み → Transformer Encoder で統合。
- Positional Encoding あり、パディング対応。最終トークンから Q 値を計算。
- Double DQN + Huber 損失、勾配クリッピング、ターゲットネット同期あり。
- リプレイバッファはシーケンス単位で保持（obs_seq, action_seq, reward_seq, next_seq, done）。
- 迷路モード `"memory"` でループ/T 字路入りのメモリ課題を確認できます。

## Ver4 (World Model + MPC) のポイント

- 履歴を統合する **HistoryEncoder (Transformer)** を共通化し、DQN とワールドモデルで同じ記憶表現を利用。
- ワールドモデルは「記憶 + 候補行動」から `next_patch (5x5) / reward / done` を教師あり学習（オンライン）。`wm_last_loss` と移動平均 `wm_avg_recent_loss` を UI に表示。
- MPC (random shooting) で horizon=5, num_samples=16, gamma=0.95 の将来行動列を評価し、最良系列の「1手目のみ」を実行。未学習時は自動で DQN にフォールバックします。
- 安定化の工夫:
  - 1手目は現在の 5x5 パッチを直接見て「確定」判定（goal / trap / wall）し、安全側のボーナス/ペナルティを即時付与。
  - goal/trap を done で混線させず、終端は安全側（罰）で評価。壁方向には小ペナルティを入れて連打停止を減らす。
- 主要ハイパーパラメータ（デフォルト）:
  - World Model: buffer_size=8000, batch_size=64, min_train_size=200, train_every=4, normalize_factor=8.0, d_model=128, nhead=4, num_layers=2, lambda_patch/reward/done=1.0。
  - MPC: horizon=5, num_samples=16, gamma=0.95, min_wm_samples=300, done_threshold=0.8, GOAL_BONUS=4.0, TRAP_PENALTY=-4.0, WALL_PENALTY=-0.75, DONE_PENALTY=-3.0, first_step_top_k=1, epsilon_mpc=0.0, softmax_temp=0.8。
  - planner_mode は `"dqn"`（TransformerベースDQN）/`"mpc"`（WM+MPC）で切替可能。

## 動作確認チェックリスト

- `uvicorn backend.app.main:app --reload` で起動し、ブラウザ `http://localhost:8000/` が開ける。
- Start/Pause/Reset/Toggle AI/Speed ボタンが動作し、ステータスが更新される。
- ランダム迷路モードで学習を進めると、成功率やエピソード報酬が徐々に向上する。
- `{"type": "maze_mode", "value": "memory"}` 送信でメモリ課題に切り替わり、学習有りの方がランダム行動より高成功率になる。
- Planner を `{"type": "planner_mode", "value": "mpc"}` で切り替え、UI の `planner_mode` 表示と WM loss が更新される（学習前は null）。
- `python -m compileall backend` がエラーなく通る。
