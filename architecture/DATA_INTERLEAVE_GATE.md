# Data interleave gate

Verified against the real SSD pool on 2026-07-31 with seed 11, sequence length
1,024, gradient accumulation 64 and source chunk size 8,192.

| Source | Manifest token weight | First 100-step observed token share |
| --- | ---: | ---: |
| local | 0.641416 | 0.661250 |
| jpnmix | 0.358584 | 0.338750 |

- both major sources first appeared in optimizer step 1;
- 95 of the first 100 optimizer steps contained both sources;
- 6,553,600 tokens were checked;
- source chunks are filled to a fixed token count, so differing document lengths
  do not distort token weights;
- the selection PRNG, per-source cursor and unread tail are checkpointed.

The manifest adapter limitation remains explicit: the completed legacy pool can
recover `local` and `jpnmix` from packed shard names, but cannot reconstruct all
original domains for each packed document.
