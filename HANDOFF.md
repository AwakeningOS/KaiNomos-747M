# 引き継ぎメモ — KaiNomos-110M

最終更新: 2026-07-29（夜間ラン投入直後）
書いた目的: **別のエージェントが、この会話の文脈なしで再開できるようにするため。**

---

## 0. 現在: 中断中（ハードウェア待ち）

**2026-07-30 00:44、訓練を停止した。再開してはいけない。** 電源ユニットを交換するまで待つ。

一晩で2回落ちた。どちらも訓練側の問題ではない。

| | 1回目 | 2回目 |
|---|---|---|
| 時刻 | 07-29 23:59 | 07-30 00:44 |
| 症状 | 即時リセット | 画面ブラックアウト＋フリーズ、強制電源断 |
| 稼働時間 | 約15分 | 約40分 |

2回目は telemetry を仕掛けてあったので、直前5分が残っている:

```
00:38:47  55°C  fan 92%  258W
   ...    （全サンプル同一）
00:43:33  55°C  fan 92%  261W   ← 最後
```

温度・ファン・電力に**変化の兆候がまったくない**まま消えた。カーネルログにも `Xid` / `NVRM` エラーが1件もない。GPU ドライバのハングなら Xid が残り、カーネルは生き続けるはずで、**カーネルごと同時停止**しているのはシステム全体が瞬断したことを示す。

除外できたもの: 熱暴走（最後まで55°C）、ファン停止（92%継続）、電力の持続超過（260W平坦）、サーマルスロットリング。

**残る第一容疑者は電源ユニット。** 根拠:
- GPU 電力は PCIe 補助電源で PSU から直接来る。マザボの VRM を通らない。CPU 単独負荷でも落ちた実績があるので、両者の共通項は PSU だけ
- i5-12600 は 65W 非K品。B660M-A の VRM が音を上げる負荷ではない
- 平均消費は約360W だが、3090 は瞬間的に2倍近く跳ねる。システム瞬間ピークは 700〜800W に達し、これが過電流保護を叩く
- PCIe 8pin は既に独立2本で配線済み（無料の対処は実施済み）
- 直近20ブート中6回が異常終了。うち3回はこのプロジェクト開始前 = 元からの持病

対処: **CORSAIR RM1000e 2025 (ATX 3.1準拠、Cybenetics GOLD、1000W)** に交換予定。ATX 3.0/3.1 は「定格200%の突入を100µs耐える」ことを要求する規格で、これが効く。

### 再開時の状態

```
runs/base_seed11/step_00000700.pt   43,008,000 token、非有限値ゼロ、optimizer完全
train.jsonl                          破損行を除去済み。train.jsonl.raw_backup が生ログ
```

`train.jsonl` には**再開リプレイによる重複 step が14件ある**。分析時は step で重複排除すること（後勝ちでよい。リプレイ値はクラッシュ前と小数第3位まで一致することを確認済み）。

再開は §1 のコマンドをそのまま実行するだけでよい。checkpoint から `data_position` も復元される。

---

## 1. いま何が動いているか

`./run_overnight.sh` がバックグラウンドで走っている。ログは `runs/overnight.log`。

| | |
|---|---|
| 内容 | Base → 10分休憩 → Adaptive、**各アーム 160M token** |
| 開始 | 2026-07-29 深夜 |
| 想定所要 | Base 3.12h + Adaptive 3.66h + 休憩 0.17h ≈ **6.95h** |
| 両アームの初期値 | `runs/kainomos_110m_init.pt`（同一） |
| 出力 | `runs/base_seed11/`, `runs/adaptive_seed11/` |

### 生存確認のしかた

```bash
cd ~/デスクトップ/KaiNomos-110M
tail -3 runs/overnight.log
tail -1 runs/base_seed11/train.jsonl | python3 -m json.tool
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

**`pgrep -f` で生死を判定しないこと。** 自分自身のシェルにマッチして「まだ動いている」と誤報する。過去に3回それをやった。PID直指定（`ps -p <pid>`）か、`train.jsonl` の最終行のタイムスタンプが進んでいるかで見る。

止めるときは `run_overnight.sh` のプロセスグループごと。単に `train` を kill すると、シェルが次のアームを始めてしまう。

### 再開のしかた

`train.py` はレジューム対応。同じ `--run-dir` に対して同じコマンドを再実行すれば、最後のチェックポイントから続く。夜間ランが途中で落ちていたら、落ちたアームのコマンドをそのまま打ち直せばよい。

---

## 2. この実験は何を測っているか

**2アームA/B。それ以外の腕は作らない。**

| アーム | 中身 |
|---|---|
| **Base** | `force_fixed=True`。固定幅・固定経路。supernet と*同じコード*を通る |
| **Adaptive** | JointRoute コントローラが実行モードを離散選択する |

判定基準はユーザーが明示している:

> **同じ計算量で NLL が下がる ⇒ 採用、下がらない ⇒ 不採用。**

同一計算量であることは `joint_cumulative_cost` が担保する。学習ログの各行にあり、`joint_budget_valid` が false になったら予算が閉じていないので、その結果は比較に使えない。

**やってはいけないこと（ユーザーが明示的に禁じた）:**
- 実装が難しいからといって KDA を別の線形注意に、MLA を素の MHA に、Latent MoE を Dense MLP に、AttnRes を素の残差に置き換える
- 目的を単一ゲートの ablation や合成タスクの診断に狭める
- Static-Matched アームや段階的パイロットを足す
- 過剰に慎重になって実験を煩雑にする（原文: 「君は過剰慎重モードにはいって実験を煩雑にする、気をつけて」）

---

## 3. GPU の運用ルール

- **連続長時間稼働をさせない。** 1サイクル = 100M token 相当のラン1本、その後 10分休憩。
- 24GB。micro-batch 6 が上限（実測ピーク Base 17.9GB / Adaptive 18.7GB）。**mb=8 は 23.2GB まで行くので使わない。**

実測スループット（110M, mb=6, seq 1024）:

| アーム | tok/s | ピーク VRAM | 1.97B 1周 |
|---|---|---|---|
| Base | 14,242 | 17.9 GB | 38.4h |
| Adaptive | 12,139 | 18.7 GB | 45.1h |

トークン数を決めるときは**時計ではなくトークンで揃える**。同じ時刻で両方止めると、速い Base だけが多くデータを見てしまい、比較が壊れる。

---

## 4. データとトークナイザ

`data/pool/manifest.json` が正。

```
KaiNomos-DataMix-v1     1,988,270,624 tokens
  train      1,972,436,918
  validation     7,977,926
  test           7,855,780
比率 ja 0.75 / en 0.10 / code 0.10 / math 0.05  （設計値ちょうど）
tokenizer  data/tokenizer/kainomos.model
           SentencePiece Unigram, vocab 32,768, byte fallback
           sha256 8b5cddc8...
```

- 名前に 2.5B や 2B と書かないこと。実測 1.99B。ja_web が想定 4.5 bytes/token に対し**実測 5.65** だったため、設計比率を厳密に保ったまま最も枯渇したソースで止めた。比率を崩して数字を丸めるより、小さくても設計どおりのプールのほうが価値がある。
- `train.py` は語彙サイズを**プールの manifest から読む**。モデル既定値からではない。トークナイザが変わればデータも変わるので、この2つがずれてはいけない。`tie_word_embeddings` は訓練時に True 固定。

### 汚染除去について（触る前に読むこと）

3回のユーザー指摘で作り直した経緯がある。安易に単純化しないこと。

1. `llm-book/aio-passages` は **NIILC ではない**。AI王用の Wikipedia 検索コーパスであり、汚染インデックスに入れると日本語 Wikipedia が丸ごと消える。NIILC は公式 dev/test XML の質問・回答のみから作る。
2. 照合単位は **canonical record**。短いレコードは全体の部分文字列一致で見る。**単一 n-gram 一致だけでは削除しない。**
3. 主判定は **評価レコードのカバレッジ**であり、ヒットした n-gram の個数ではない。個数は文書長に比例するので、長い文書を残して短い問題を見逃す（＝ちょうど逆）。`distinct >= 50` を全ベンチ共通ルールにしてはいけない。ルールは `contamination_match.py` の `RULES` にベンチマークごとに書いてある。

自分で見つけた落とし穴も記録しておく:
- 英単語 13-gram は日本語で必ず 0 件になる。文字 n-gram（20文字）と併用が必要。
- 言語判定を文書単位でやると、日本語ページに埋め込まれた英語ベンチマークが見えない（gsm8k/humaneval/mbpp が 0/40）。
- 修正後の陽性コントロール: JCommonsenseQA 3/40 → **40/40**、GSM8K/HumanEval/MBPP 0/40 → **40/40**、数学の偽陽性 422 → 16。

削除は**物理削除ではなく復元可能な隔離**。

---

## 5. 移植（`runs/kainomos_110m_init.pt`）

82M Base から 110M へ。移植元は train loss ではなく、**82M Base と Joint++ の同一 validation/test NLL** で選んだ（Base 4.0321 / 4.0349 が勝ち、Joint++ は 4.1212 / 4.1226 で不採用）。

```
source layers   : 13 -> 16
tensors copied  : 145
shape mismatch  : 54
fresh layers    : [13, 14, 15]
new tensors     : 286
vocabulary      : special_or_empty 259, exact_piece 1921,
                  decomposed_mean 30588, total 32768
total params    : 111,042,670
```

**名前・形状・役割がすべて一致するものだけコピーする。** 増えた深さ、広げた FFN チャネル、MUDD、Delta、コントローラ、MTP は、別の意味を持っていた重みを流用せず自前の初期化を保つ。旧 LM head は消えた語彙の上の別テンソルなので引き継がず、新 embedding に tie する。

---

## 6. 過去に踏んだ地雷（同じ穴に落ちないために）

| 症状 | 原因 | 対処 |
|---|---|---|
| backward が壊れる | MoE dispatch の `torch.argsort` が不安定ソート。gradient checkpoint の再計算でエキスパート区間長が変わる | `stable=True` |
| checkpoint 再計算で候補数が 3→4 | 生きたリストを `torch.utils.checkpoint` に渡していた | `list(tracker.candidates)` でスナップショット |
| KDA が 80.6倍遅い（5.3→424.7ms） | マスク経路が Python ループに落ちていた | DECAY_ONLY を β=0 として表現（→1.13倍） |
| force_fixed が標準モデルと一致しない | (a) マスクをスキップして 2816 幅を実行 (b) 2816 からのスライスは native init と別物 (c) マスクと slice で総和順序が違い 2e-6 差 (d) ALL 深さ tier が tiered softmax を通っていた | 常に決定を実行 / state_dict で明示コピー / `uniform_width` slice 経路 / 素の softmax 経路 |
| force_fixed のコストが目標から 2.8% ずれる | コストモデルの変数シャドーイング。MUDD は過去*層*、Delta は過去*ブロック*、両方 `sources` という名前で、Delta の value 読み出しが MUDD の本数で課金されていた | 別名に分離 |
| eval 時に予算 5.3% 超過 | サンプルされた方策で予算を閉じ、デプロイされる方策で閉じていなかった | straight-through argmax で訓練＋積分補正を撤去（train/eval 差 5.28% → 0.0000%） |
| Adaptive が 0.7% 損をする | 目標が素の 1.0 だったが Base は実測 1.00694 かかっていた | `budget_target = budget_ratio * full_policy_cost` |
| code 比率が 8.0% に落ちる | 10% を8言語で均等分割していた（設計シェア無視）。Python が 107,072 件中 22,480 件で頭打ち | `SOURCE_SHARE` で設計シェアを明示 |
| `train.idx.npy` ができる | `np.save` が拡張子を足す | ファイルハンドルを明示的に渡す |
| RNG state のロードに失敗 | CUDA テンソルとして読んでいた | `map_location="cpu"` |

`trust_remote_code=True` は使わない（parquet ブランチのみ）。

---

## 7. 次にやること

1. 朝、両アームの 160M 完了を確認する。`runs/*/train.jsonl` の最終行で `joint_budget_valid: true` と `joint_cumulative_cost` が目標近傍にあることを見る。
2. **同一トークン数のチェックポイントで** validation NLL を比較する（`eval.py`）。これが採否の判定そのもの。
3. ユーザーの最終目標は「過剰訓練にならずに、一番モデルの性能があがるポイント」。1.97B を何周させるかはまだ決まっていない。validation NLL の推移を見て決める。全周させると Base 38.4h / Adaptive 45.1h なので、一気にやらず夜間セグメントを積み重ねる。

夜間セグメントを追加で回すには:

```bash
TOKENS=160000000 ./run_overnight.sh    # 前回の続きから
```

---

## 8. ユーザーとの仕事のしかた

- **断定して進める。** 曖昧にぼかしたり、実験の提案だけして実行しない、をしない。コードを走らせて検証する。
- 実験を細分化しない。いきなり完全版を動かして、性能が上がるか上がらないかだけ見る。
- 結果は正直に。落ちたら落ちたと出力つきで言う。飛ばした手順は飛ばしたと言う。
- 旧世代のフォルダ（`mini_kimi_organism/`, `K3Mini-82M/`）は**消さずに保存**。結果は確定済み。
- 名前について: 対外的には **KaiNomos-110M**、アームは **Base / Adaptive**。「+++」表記を使わない。Kimi K3 を名前に入れない。タグラインは「Reforming the laws of compute allocation. / 計算資源配分の法則を作り変える。」内部コード名は壊れないよう段階的に変える。
