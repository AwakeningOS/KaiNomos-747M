# KaiNomos-750M

KaiNomos-750Mは、24GBのコンシューマーGPU 1枚で事前学習できることを目標に
設計した、日本語中心のdecoder-only言語モデルです。単にモデルを小さくする
のではなく、限られた計算量の中で「近い文脈を細かく覚える処理」と「離れた
文脈を見渡す処理」をどう組み合わせるかに重点を置いています。

現在公開しているのは、学習前のアーキテクチャと訓練基盤です。学習済み重みや
性能ベンチマークはまだ公開していません。

## モデルが文章を処理する仕組み

モデル本体は24層で、4層を1組としたstageを6回繰り返します。

```text
(KDA → KDA → KDA → MLA) × 6
```

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

## 検証方針

Delta Blockの効果は、同じ初期値・tokenizer・data order・optimizer・token予算を使う
次の2条件で比較します。

- 通常残差だけのbaseline
- Delta Blockを有効にしたKaiNomos-750M

training lossだけでは選ばず、固定held-out splitのnext-token NLLで判断します。MTPの
有無はarchitecture決定後に別途比較します。

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
```

## Tests

```bash
cd architecture
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
python -m ruff check .
```

CPU試験ではparameter数、因果性、文書分離、cache一致、Delta source、Muon分類、data
resume、checkpoint整合性を確認しています。CUDA BF16とFLA kernelの実機検証、学習
結果の評価はCPU試験とは分けて記録します。

## Status

- architecture実装：完了
- CPU correctness tests：合格
- deterministic interleave / exact resume：合格
- CUDA BF16 / FLA acceptance：未完了
- 学習済みweights：未公開
- downstream benchmark：未実施

実装試験の合格は、言語モデルとしての性能を示すものではありません。

## License and attribution

プロジェクト独自コードはApache License 2.0です。KDA、MLA、Delta Block、Muon、MTP
は公開論文に基づく既知機構であり、それらを組み合わせたこと自体を新規研究成果とは
主張しません。詳細は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照して
ください。
