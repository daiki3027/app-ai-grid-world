# GridWorld DQN Demo

FastAPI + WebSocket で動くシンプルな GridWorld 学習デモです。CNN を使った DQN で、ブラウザから学習の進み具合をリアルタイムに確認できます（ランダム迷路に対応）。

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
- WebSocket に `{"type": "maze_mode", "value": false}` を送ると固定迷路モードに戻ります（デフォルトはランダム迷路）。

ステータス欄で `episode`, `step`, `epsilon`, `total_reward`, 直近エピソードの成功率などがリアルタイムに更新され、Canvas 上でエージェントの動きが見られます。学習が進むほどゴールまでの経路が短くなる様子が確認できます。

## 仕様メモ

- GridWorld (10x10) に壁・スタート・ゴール・罠を配置。ランダム迷路は必ず「罠を踏まないスタート→ゴール経路」が存在するよう生成・検証しています。
- 観測はグリッド + エージェントを 8 で描画した「疑似画像」。学習入力はエージェント中心の 5x5 パッチ（壁外は 1 埋め）を CNN へ入力。
- 報酬: ゴール +1, ステップ -0.01, 壁衝突 -0.05, 罠 -1 (終了) に加え、最短距離が縮むとわずかに加点（デフォルト ON）。
- 1 エピソード最大 200 ステップ。エージェントは DQN (CNN + experience replay + target network) で、ε-greedy / ε 減衰。
- WebSocket で各ステップの状態を配信 (grid, agent_pos, episode/step, reward/total_reward, epsilon, success_rate, done など)。追加で `observation_patch`, `q_values`, `random_maze` なども送信します。

乱数シードはデフォルトで固定済みです。
