# GridWorld Q-learning Demo

FastAPI + WebSocket で動くシンプルな GridWorld の Q-learning 学習デモです。ブラウザで学習の進み具合をリアルタイムに確認できます。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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

ステータス欄で `episode`, `step`, `epsilon`, `total_reward`, 直近エピソードの成功率などがリアルタイムに更新され、Canvas 上でエージェントの動きが見られます。学習が進むほどゴールまでの経路が短くなる様子が確認できます。

## 仕様メモ

- GridWorld (10x10) に壁・スタート・ゴール・罠を配置。
- 報酬: ゴール +1, ステップ -0.01, 壁衝突 -0.05, 罠 -1 (終了)。
- 1 エピソード最大 200 ステップ。エージェントは Q-learning (表形式) を使用し、ε-greedy で行動・ε は逐次減衰。
- WebSocket で各ステップの状態を配信 (grid, agent_pos, episode/step, reward/total_reward, epsilon, success_rate, done など)。

乱数シードはデフォルトで固定済みです。
