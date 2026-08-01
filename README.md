# KaiNomos-750M

**日本語** | [English](README_EN.md)

KaiNomos-750Mは、24GBのコンシューマーGPU 1枚で事前学習できることを目標に
設計した、日本語中心のdecoder-only言語モデルです。単にモデルを小さくする
のではなく、限られた計算量の中で「近い文脈を細かく覚える処理」と「離れた
文脈を見渡す処理」をどう組み合わせるかに重点を置いています。

このrepositoryでは、モデル実装、厳密resume対応の訓練基盤、RTX 3090向けruntimeを
公開しています。49,152語彙の対応トークナイザー、データpacker、訓練・resume、
checkpoint対話CLIまで同梱しています。学習済み重みはまだ公開していません。

> **CUDA GPUが必要です。** 公開Quick Startはfull 718M backboneをGPUで動かします。
> 実測と最適化の基準環境は24GBのRTX 3090です。

## Quick start

```bash
git clone https://github.com/AwakeningOS/KaiNomos-750M.git
cd KaiNomos-750M
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python examples/quickstart.py
```

最後のコマンドはfull 718M backboneをCUDAで動かし、logitsのshape、parameter数、
Delta routing記録数を表示します。学習済みモデルの文章生成ではなく、実装が正しく
GPU上でforwardできることを確認する例です。測定環境の詳細は
[ENVIRONMENT_LOCK.md](ENVIRONMENT_LOCK.md)にあります。

## 自分のデータで訓練して対話する

対応トークナイザーは同梱済みです。train/validationデータを用意して実行します。

```bash
python tools/prepare_data.py --input corpus/train.jsonl --output data/myrun --split train --source-id mydata
python tools/prepare_data.py --input corpus/validation.jsonl --output data/myrun --split validation --source-id mydata
python scripts/run_kainomos_runtime_tuned.py --runtime-activation-checkpointing on --runtime-micro-batch 16 --runtime-checkpoint-every-steps 200 --architecture kainomos_750m_v1 --data-dir data/myrun --run-dir runs/myrun --device cuda --allow-gpu --optimizer muon --depth-routing delta_block --mtp off --target-tokens 65536
```

同じ訓練コマンドを再実行すると`latest.json`からresumeします。保存後は次で対話できます。

```bash
python examples/chat.py --checkpoint runs/myrun/step_00000001.pt
```

入力形式と複数sourceの混合は[TRAINING.md](TRAINING.md)にまとめています。
生成時はpromptをEOD区間ごとに一括prefillし、以降はKDA stateとMLA latent cacheを
使って新しい1 tokenだけを入力します。実測と不採用のMLA吸収decodeは
[INFERENCE.md](INFERENCE.md)に記録しています。

## モデルが文章を処理する仕組み

モデル本体は24層で、4層を1組としたstageを6回繰り返します。

```text
(KDA → KDA → KDA → MLA) × 6
```

各stageの処理を概念的に書くと次の形です。

```text
h = token_embedding
completed_deltas = []

for stage in 6 stages:
    stage_input = h
    for attention in [KDA, KDA, KDA, MLA]:
        context = DeltaRoute(h, token_embedding, completed_deltas)
        h = h + attention(RMSNorm(context))

        context = DeltaRoute(h, token_embedding, completed_deltas)
        h = h + SiTU_GLU(RMSNorm(context))

    completed_deltas.append(h - stage_input)
```

DeltaRouteは参照用のcontextを作りますが、`h`の主残差そのものは置き換えません。

最初の3層に置いたKDAは、過去のtokenを固定長のKV cacheとして保存する代わりに、
recurrent stateへ順次書き込みます。直前の語や局所的な変化を追いながら、文脈長に
比例してcacheが増えないのが特徴です。

4層目のMLAは、そのstageの情報をより広い範囲で読み直す役割を持ちます。Keyと
Valueをそのまま保存せず、256次元のlatentへ圧縮するため、通常のattentionより
cacheを小さくできます。位置埋め込みには依存せず、QueryとKeyはhead単位で正規化
します。

つまり、KDAが文章の流れを逐次追跡し、MLAが一定間隔で文脈全体を整理する構成です。
最終層もMLAなので、出力直前には必ずglobal attentionを通ります。

## 深さ方向の情報をどう使うか

通常のTransformerでは、各層の出力を順番に残差へ足していきます。しかし深い層から
見ると、「以前の層が何を変えたのか」と「それまでに蓄積された状態」が混ざって
しまいます。

KaiNomos-750Mでは、4層のstageが入力をどれだけ変化させたかを
`stage_output - stage_input`として保存します。後段はDelta Blockを通じて、埋め込みと
各stageが生んだ変化を選んで参照できます。主残差はそのまま維持され、routing結果は
追加情報として加算されます。

最初のattentionには参照可能な過去の変化がないため、routingを行わず厳密なidentity
として動きます。また、役割が重複していたMuDDは削除し、深さ方向の機構をDelta
Blockへ一本化しています。

## FFNと学習時の補助機構

各層のFFNには、gate側へ`SiLU(x) × tanh(softplus(x))`を用いるSiTU-GLUを採用して
います。FFNは疎なexpert方式ではなく、すべてのtokenが同じdense FFNを通ります。

MTP（Multi-Token Prediction）も実装していますが、既定では無効です。通常の
next-token predictionに本当に役立つかを別の比較実験で確認し、held-out NLLが改善した
場合だけ本学習に採用します。MTP自身のlossが下がっただけでは採用しません。

## モデル規模

| 項目 | 値 |
| --- | ---: |
| 層数 | 24 |
| hidden size | 1,280 |
| dense FFN | 5,120 |
| attention heads | 10 |
| head dimension | 128 |
| 語彙数 | 49,152 |
| 学習context | 1,024 token |
| 配備backbone | 718,341,812 parameters |
| 学習専用MTP | 31,491,978 parameters |
| MTP込み訓練モデル | 749,833,790 parameters |

EmbeddingとLM headは重みを共有します。KDAは18層、MLAは6層です。

## Optimizer

大きな行列にはMuonを使い、KDAとMLAのQ/K/V projectionはheadごとに独立して更新
方向を整えます。Embedding、Norm、bias、decay parameterなどはAdamW側で扱います。
MuonとAdamWのlearning rateは、RMSを揃えた上で共通の`0.0003`を使用します。

parameterの分類は名前と役割で明示しており、新しいparameterが未分類のまま追加
されると訓練を拒否します。

## RTX 3090向けruntime効率化

モデル、optimizer、data order、65,536 token/stepを変えず、peak reserved VRAM
`22.0 GiB`以下をhard gateとしてruntimeを最適化しました。

| 構成 | steady tok/s | peak reserved |
| --- | ---: | ---: |
| mb8 stage-2 bracket baseline | 2,773.20 | 19.840 GiB |
| mb8 全FLA RMSNorm＋Delta score | 3,518.97 | 17.285 GiB |
| 採用mb16、10-step確認 | **3,587.86** | **21.957 GiB** |

採用構成はactivation checkpointing ON、micro batch 16 / accumulation 4、
chunked CE 32、BF16 variable-length Flash MLA、全FLA BF16 RMSNorm、fused Delta
score、`expandable_segments:True`です。bracket baseline比で29.38%高速、同じ平均電力
ならtoken当たりの時間・GPU電力量を22.71%削減します。

詳細な成功・失敗候補、比較条件、fallbackは[EFFICIENCY.md](EFFICIENCY.md)を参照して
ください。これはruntime測定であり、言語性能の改善を示す結果ではありません。

## 文書境界とデータ供給

複数文書を1系列へ詰めても、前の文書が次の文書へ漏れないようにしています。

- MLAのattention mask
- KDAのrecurrent state
- KDAの短いconvolution state
- next-token loss
- MTP loss

これらをEOD token（ID 4）で同時に区切ります。

学習データはsourceごとに独立したcursorを持ち、固定token長のchunkをseed付きで混合
します。checkpointにはdata cursor、未消費のread-ahead、optimizer、Python/NumPy/
PyTorch/CUDAのRNG stateを保存するため、中断後も同じtoken列から再開できます。

## 既定の学習構成

公開実装の既定値はDelta Block有効、MTP無効です。対応トークナイザーは
`tokenizer/kainomos-49152.model`として同梱しています。学習済みweightは未公開です。

## Repository layout

```text
architecture/
├── model.py          # 24層KDA/MLA backbone
├── kda.py            # recurrent KDAとFLA path
├── mla.py            # NoPE MLAとlatent cache
├── delta_block.py    # 深さ方向のDelta routing
├── muon.py           # Per-Head Muon / AdamW分類
├── interleave.py     # source混合と厳密resume
├── train.py          # 訓練、checkpoint、validation
├── observe.py        # architecture/optimizer観測
└── tests/            # CPU correctness tests
scripts/
├── run_kainomos_runtime_tuned.py       # 採用runtimeで厳密resume
├── kainomos_optimization_runtime.py    # state-dictを変えないruntime patch
├── benchmark_kainomos_runtime_candidate.py
├── benchmark_kainomos_generation.py    # prefill/decode A/B
└── validate_optimization_cuda.py       # CUDA parity gate
examples/
├── quickstart.py       # full CUDA forwardの最小例
└── chat.py             # checkpoint対話CLI
tokenizer/
└── kainomos-49152.model # 同梱SentencePiece tokenizer
tools/
└── prepare_data.py     # JSONL/textから学習shardを作成
```

## Tests

```bash
cd architecture
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
python -m ruff check .
```

CPU試験ではparameter数、因果性、文書分離、cache一致、Delta source、Muon分類、data
resume、checkpoint整合性を確認しています。CUDA BF16/FLAの同値性と22 GiB上限内の
実測も完了しています。学習結果の評価は実装試験・速度試験とは分けて記録します。

## Status

- architecture実装：完了
- CPU correctness tests：合格
- deterministic interleave / exact resume：合格
- CUDA BF16 / FLA runtime同値性：合格
- 採用runtime：3,587.86 tok/s、peak reserved 21.957 GiB（10-step確認）
- 学習済みweights：未公開
- downstream benchmark：未実施

runtimeの使い方と速度検証の詳細は[EFFICIENCY.md](EFFICIENCY.md)を参照してください。

実装試験の合格は、言語モデルとしての性能を示すものではありません。

## License and attribution

プロジェクト独自コードはApache License 2.0です。KDA、MLA、Delta Block、Muon、MTP
は公開論文に基づく既知機構であり、それらを組み合わせたこと自体を新規研究成果とは
主張しません。詳細は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照して
ください。
