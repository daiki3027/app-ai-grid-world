# GridWorld RL Demo (Transformer Memory + World Model + MPC)

FastAPI + WebSocket で動く GridWorld の学習デモです。**Transformer で観測履歴を記憶する DQN** をベースに、オプションで **ワールドモデル + MPC** による先読み計画も試せます。ブラウザから学習の進み具合をリアルタイムに確認できます（ランダム迷路 / 固定迷路 / メモリ課題）。

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

- **Start / Pause / Reset / Toggle AI / Speed 1x/5x/20x**: 学習開始・一時停止・初期化・学習ON/OFF切替・速度変更。
- **Planner: DQN / MPC**: 記憶付きDQNのみか、ワールドモデルを使ったMPCで先読みするかを切替。
- WebSocket でモード切替:
  - `{"type": "maze_mode", "value": false}` : 固定迷路
  - `{"type": "maze_mode", "value": true}` : ランダム迷路（デフォルト）
  - `{"type": "maze_mode", "value": "memory"}` : 記憶が必要なループ/T 字路迷路
  - `{"type": "planner_mode", "value": "mpc"}` / `"dqn"` : プランナー切替

ステータス欄で `episode`, `step`, `epsilon`, `total_reward`, `success_rate`, `learning` に加え、`planner_mode` / `planner_used`、`last_loss` / `avg_recent_loss`、`wm_last_loss` / `wm_avg_recent_loss`、`wm_samples_needed` / `wm_buffer_size` などがリアルタイム更新されます。Canvas 上でエージェントの動きを視覚的に確認できます。

## 仕様メモ

- **環境**: 10x10 GridWorld（壁・スタート・ゴール・罠）。ランダム迷路は必ずスタート→ゴールの安全経路が存在するよう生成。
- **観測**: グリッドを「疑似画像」として扱い、エージェント中心の 5x5 パッチ（境界外は壁扱い）を入力。
- **報酬**: ゴール +1, ステップ -0.01, 壁 -0.05, 罠 -1（終了）に加え、ゴール最短距離が縮むとわずかに加点（デフォルト ON）。
- **DQN (Transformer 記憶付き)**: 過去 16 ステップの「5x5 観測 + 行動 + 報酬」をトークン化し、CNN + Transformer Encoder で統合した記憶から Q 値を計算。Double DQN + Huber 損失 + ターゲットネット同期。
- **ワールドモデル**: 記憶 + 候補行動から「次パッチ/報酬/終了確率」をオンライン学習し、`wm_last_loss` / `wm_avg_recent_loss` を UI に表示。
- **MPC (先読み計画)**: horizon=5, num_samples=16, gamma=0.95 で行動列をシミュレーションし、最良系列の「最初の1手」を実行。1手目は現在の5x5パッチを直接見て安全判定（goal/trap/wall）。done予測は閾値0.6で終端扱い、未学習時は自動で DQN にフォールバック。
- **主要ハイパーパラメータ（デフォルト）**:
  - World Model: buffer_size=8000, batch_size=64, min_train_size=200, train_every=4, normalize_factor=8.0, d_model=128, nhead=4, num_layers=2, lambda_patch/reward/done=1.0
  - MPC: horizon=5, num_samples=16, gamma=0.95, min_wm_samples=300, done_threshold=0.6, first_step_top_k=1, epsilon_mpc=0.0, softmax_temp=0.8, GOAL_BONUS=4.0, TRAP_PENALTY=-4.0, WALL_PENALTY=-0.75, DONE_PENALTY=-3.0
- **通信**: WebSocket で各ステップの状態を配信（grid, agent_pos, episode/step, reward/total_reward, epsilon, success_rate, done, observation_patch, q_values, planner_mode/used など）。

## 動作確認チェックリスト

- `uvicorn backend.app.main:app --reload` で起動し、ブラウザ `http://localhost:8000/` が開ける。
- Start/Pause/Reset/Toggle AI/Speed ボタンが動作し、ステータスが更新される。
- ランダム迷路モードで学習を進めると、成功率やエピソード報酬が徐々に向上する。
- `{"type": "maze_mode", "value": "memory"}` 送信でメモリ課題に切り替わり、学習有りの方がランダム行動より高成功率になる。
- Planner を `{"type": "planner_mode", "value": "mpc"}` で切り替え、`planner_mode` / `planner_used` 表示と WM loss が更新される（学習前は null）。
- `python -m compileall backend` がエラーなく通る。
