
### 1. 技術スタック

* Python 3.12
* FastAPI
* WebSocket（リアルタイム配信）
* フロントは依存ゼロの HTML + Vanilla JS（Canvas描画）
* 学習アルゴリズムは **Q-learning（テーブル）**。深層学習は不要。

### 2. アプリ仕様（必須）

* 迷路は **GridWorld**（例：10x10）。
* セル種別：空き(0)、壁(1)、開始S、ゴールG、罠T（任意だができれば入れる）。
* AIは毎ステップで行動（上/下/左/右）。壁は通れない。
* 報酬設計：

  * ゴール到達：+1
  * 1ステップごと：-0.01（短経路を促す）
  * 壁にぶつかる（移動失敗）：-0.05
  * 罠（任意）：-1 でエピソード終了
* 1エピソードの最大ステップ数を設定（例：200）。超えたら終了。

### 3. “進化が見える”UI（必須）

* ブラウザで表示（localhost）。
* Canvasで盤面を描画し、AIの位置をアニメーション表示。
* UIに以下を表示：

  * episode、step、epsilon、total_reward、success_rate（直近N回）
* ボタン：

  * Start（学習開始）
  * Pause（一時停止）
  * Reset（Qテーブル初期化）
  * Toggle AI（学習ON/OFF。OFFのときはランダム行動でも良い）
* スピード調整（例：1x/5x/20x）。内部ループで間引きでもOK。

### 4. バックエンド要件

* WebSocketでフロントへ、毎ステップ or 間引きで状態を配信：

  * grid（2D配列でも可）
  * agent_pos
  * episode/step
  * reward/total_reward
  * epsilon
  * done
  * last_action
* Q-learning：

  * 状態は (x, y) の離散状態でOK
  * 行動は4つ
  * α（学習率）、γ（割引率）、ε（探索率）を設定。εは減衰させる（例：0.995）
* 乱数seedは固定できるように（再現性）。

### 5. プロジェクト構成

* backend/

  * app/main.py（FastAPI起動、WSエンドポイント、静的配信もここでOK）
  * app/gridworld.py（環境）
  * app/qlearning.py（エージェント）
* frontend/

  * index.html
  * app.js
  * style.css
* 起動手順を README.md に書くこと。

### 6. 実装方針

* まず「最小で動く」ことを優先。
* 外部ライブラリの追加は最小限（FastAPIとuvicorn程度）。
* すべてローカルで動く。DB不要。
* 迷路は固定でも良いが、できれば「迷路を数パターン切り替え」できると嬉しい（オプション）。

### 7. 生成してほしい成果物

* 必要な全ファイルを作成し、内容を実装してください。
* `requirements.txt` を作成。
* `README.md` に以下を含める：

  * セットアップ（venv、pip install）
  * 起動コマンド（uvicorn）
  * ブラウザURL
  * 期待される挙動（学習が進むと最短経路に近づく）

### 8. 追加の品質条件

* 例外で落ちない（WS切断時に安全に停止）
* 可能なら、学習ループはバックグラウンドタスクとして実装（asyncio）
* コードは読みやすく、関数/クラスに責務を分ける

---
