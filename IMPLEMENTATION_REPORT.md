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

`migrate_82m_to_110m.py`。

- 既存13層 → 同種の層へそのままコピー（KDA と MLA を跨いだコピーは禁止し、テストで固定）
- FFN 2432→2816：既存を完全コピー、追加channelは新規初期化のうえ down_proj を 0.1 倍
- 新規3層：同種の後半層から循環コピーし、attention/FFN の出力projectionを 0.1 倍
- MUDD / Delta / controller / MTP は各自の identity 初期化を保持

## 未確定・次の作業

- 初期 checkpoint は 82M Base と Adaptive 版の完走後、validation NLL が低い方に決める
- tokenizer は 32,768 SentencePiece Unigram + weight tying へ交換（測定で決定、下記）
- 本番データは `KaiNomos-DataMix-2.5B-v1`（`data_mix.py`）

### tokenizer 交換の根拠（実測）

現行 tokenizer は FineWeb-Edu（英語のみ）で訓練した 16,384 語彙 ByteLevel BPE で、
日本語が **2.56 tokens/文字** とバイト列へ崩壊する。

```
「日本語の」→ ['æ','Ĺ','¥','æ','ľ','¬','è','ª','ŀ','ãģ','®','ã']
```

16,384 untied → 32,768 tied は総パラメータが同一（16,384×512×2 = 32,768×512 =
16,777,216）なので、語彙を倍にしても 110M 設計の再調整は不要。
