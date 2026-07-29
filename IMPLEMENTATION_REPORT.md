# K3Mini-110M 実装報告

## パラメータ

| 項目 | 値 |
|---|---|
| 総パラメータ | **111,042,670** |
| 学習可能 | 111,042,670 |
| 推論時（MTP除く） | 106,409,510 |
| MTP専用 | 4,633,160 |
| controller | 39,694 |
| MUDD-QKV | 583,576 |
| Delta | 166,432 |

目標範囲 109,000,000–112,000,000 に収まっており、MTP block の FFN 幅は指示書既定の
1792 のまま変更していない。

## 構成

- 16層 `KKKM KKKM KKKM KKKM`、d_model 512、層スキップなし
- Nested Dense FFN 1024/1408/1792/2176/2432/2816（単一の SiTU-GLU 行列の先頭channel）
- MUDD-QKV：全過去層から Q/K/V を個別合成、identity 初期化
- Projected Low-Rank Delta Block：value 512次元、routing key 64次元、gate=0 開始
- MTP-1：`(h_t, E(y_t+1)) -> y_t+2`、専用 KDA block、weight 0.30、step 0 から有効
- JointRoute：K/M/F/R をひとつの共通価格で配分、予算は16層固定モデルと同一

## GPU smoke test

| 項目 | 値 |
|---|---|
| micro-batch × seq | 2 × 1024（grad accum 32） |
| tokens/sec | 3,571 |
| peak VRAM | 7.72 GB |
| 初期 NTP loss | 9.881 |
| 初期 MTP loss | 9.8708 |
| 実測 compute ratio | **1.0** |
| checkpoint save/reload | OK |

forward → backward → optimizer.step → evaluation forward → checkpoint save →
reload まで1回通っている。上表は GPU が 82M の訓練で占有されていたため
micro-batch 2 で測った値で、本番 micro-batch 8 の数値は 82M 完走後に再測する。

### 初期 route 使用率

```
{
  "K": [
    0.0,
    3.0
  ],
  "F": [
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0
  ],
  "R": [
    0.0,
    0.0,
    0.0,
    1.0
  ],
  "M": [
    0.0,
    1.0
  ]
}
```

K=FULL / M=GLOBAL_READ / F=1792 / R=ALL、すなわち Base policy と完全一致で開始し、
実測 compute ratio も 1.000000 だった。JointRoute が Base を厳密に包含している。

## CPU テスト

26 passed。内訳は shape・causality・MUDD identity・Delta identity・budget・
checkpoint migration。

特に次を固定している。

- 未来token・未来層・未来blockを参照しない
- MUDD identity 初期化で出力が最新 depth state と一致する
- Delta gain=0 で残差が完全に不変
- FFN 幅の包含関係
- MTP の target shift が `t -> t+2`（渡された t+1 を予測しない）
- force_fixed のコストが budget target と一致

## 実装中に見つけて直した欠陥

**cost model の変数シャドウイング。** MUDD は過去*層*を、Delta は過去*ブロック*を
見るが、同じ `sources` という名前を再代入していたため、Delta の value read に
MUDD の source 数が使われていた。force_fixed のコストが目標と 2.8% ずれて発覚。

## 移植

`migrate_82m_to_110m.py`（語彙変換の実装本体は `migrate_vocab.py`）。

- 82M checkpointから、名前・形状・役割が一致する既存層だけをコピー
- KDAとMLAを跨いだコピー、形状の違うFFNの部分コピー、新規層へのcloneは禁止
- 追加3層 / MUDD-QKV / Delta Block / controller / MTPは110M側の新規初期化を保持
- 旧語彙との完全一致は旧input embeddingをコピー
- 旧token列へ可逆に分解できる新tokenは、対応する旧input embeddingの平均で初期化
- それ以外の新語彙行は通常初期化
- LM headは新embeddingとweight tyingし、旧LM headは独立tensorとして移植しない
- JointRouteの機構・設定・経路選択ロジックは変更しない

## 未確定・次の作業

- 移植元は82M BaseとJoint++を同一validation/test splitで評価し、NLLが低い方に決める。train lossだけでは選ばない
- tokenizerはKaiNomos-110M専用の32,768 SentencePiece Unigram + weight tying。82M側は変更しない
- 本番データは `KaiNomos-DataMix-v1`、正確には1,988,270,624 tokens（`data/pool/manifest.json`）

### tokenizer 交換の根拠（実測）

現行 tokenizer は FineWeb-Edu（英語のみ）で訓練した 16,384 語彙 ByteLevel BPE で、
日本語が **2.56 tokens/文字** とバイト列へ崩壊する。

```
「日本語の」→ ['æ','Ĺ','¥','æ','ľ','¬','è','ª','ŀ','ãģ','®','ã']
```

16,384 untied → 32,768 tied は総パラメータが同一（16,384×512×2 = 32,768×512 =
16,777,216）なので、語彙を倍にしても 110M 設計の再調整は不要。
