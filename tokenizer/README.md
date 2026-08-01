# KaiNomos tokenizer

KaiNomos-750M uses the included SentencePiece tokenizer without modification.
Its vocabulary size and special-token IDs are part of the model/checkpoint
contract; substituting another tokenizer makes existing data shards and
checkpoints incompatible.

| Item | Value |
| --- | --- |
| Vocabulary size | 49,152 |
| Unknown | ID 0, `<unk>` |
| Padding | ID 1, `<|pad|>` |
| Beginning of sequence | ID 2, `<|bos|>` |
| End of sequence | ID 3, `<|eos|>` |
| End of document | ID 4, `<|eod|>` |
| Model SHA-256 | `dcbb5054b28539d82243d3ee66930b244e876b01f087db607906c445be18f695` |
| Vocab SHA-256 | `45227401fdcb08d65acc06f5247bbd67f9464b16630592e911ad021a4c6ce8d9` |

Use `kainomos-49152.model` for tokenization. The `.vocab` file is a readable
piece inventory; training and inference do not require it. The corpus used to
build the vocabulary is not included. Users remain responsible for ensuring
that their own training data may legally be used for model training.
