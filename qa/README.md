# Abstractive QA experiments

The trainer is a common multilingual mT5 seq2seq adaptation of the five
abstractive papers. Each method is represented by its paper-inspired input
serialization and control markers; the original papers' bespoke selector,
pointer-generator, gate, or reasoning-decoder architectures are not claimed
to be reproduced exactly.

For a one-command or one-click run from the repository root:

```bash
python -m pip install -r requirements-qa.txt
./run_qa.sh
```

The launcher prepares missing data, trains `pathfid` on `true_multihop`, and
evaluates the dev split with greedy decoding. Override settings, for example:

```bash
METHOD=msg MODEL_NAME=google/mt5-small MAX_STEPS=1 MAX_TRAIN_EXAMPLES=8 ./run_qa.sh
```

Run the four requested Vietnamese/multilingual backbones with identical QA
settings:

```bash
./run_qa_all_models.sh
```

The presets are `VietAI/vit5-base`, `vinai/bartpho-syllable-base`,
`google/mt5-base`, and `facebook/mbart-large-50-many-to-many-mmt`. mBART is
configured with Vietnamese `vi_VN` source/target codes and forced Vietnamese
generation.

Training checkpoints are saved every 50 optimizer steps under
`runs/qa/<track>_<method>_seed42/checkpoints`. If the process is interrupted,
running `./run_qa.sh` again automatically resumes from the latest checkpoint.
After successful completion, the `COMPLETE` marker prevents accidental
retraining. Use `SAVE_STEPS=10 ./run_qa.sh` to checkpoint more frequently.

Prepare the QA tracks from structured IR data:

```bash
python -m qa.prepare_abstractive
```

This creates `all_matched` and `true_multihop` under
`dataset/QA/abstractive`. Add `--with-distractors` after installing
`requirements-qa.txt` to create BM25 candidate pools.

Run a small CPU smoke test:

```bash
python -m qa.train \
  --method pathfid \
  --track true_multihop \
  --model-name google/mt5-small \
  --max-train-examples 8 \
  --max-steps 1 \
  --output-dir /tmp/qa-runs
```

Evaluate a checkpoint:

```bash
python -m qa.evaluate \
  --checkpoint /tmp/qa-runs/true_multihop_pathfid_seed42 \
  --method pathfid \
  --data dataset/QA/abstractive/true_multihop/dev.jsonl \
  --corpus dataset/IR/structured/dev_corpus.jsonl \
  --output /tmp/qa-runs/pathfid-dev.json \
  --skip-bertscore
```

## Extractive QA experiments

Prepare weak span labels from the structured gold contexts:

```bash
python -m qa.prepare_extractive
```

Each row stores the original question/answer, at most six ordered passages of
256 regex tokens each, global `start_position`/`end_position`, and
`label_method` (`exact`, `fragment`, `subsequence`, or `best_overlap`). Passages
are selected without using the answer. The six-passage budget is allocated
round-robin across source contexts so a long first context cannot consume all
input; span labels are constrained to one passage. Full-document fallbacks are
cut to the article with the highest question-term overlap when the structured
source is available. Both `all_matched` and `true_multihop` tracks are written;
the latter requires at least two distinct positive docs and at least two
positive contexts actually represented in the model input.

Run CoG, DFGN, HGN and QANet-inspired extractive models with the shared
`xlm-roberta-base` encoder:

```bash
./run_extractive.sh
```

Prepared files are written to `dataset/QA/extractive/{all_matched,true_multihop}/{train,dev,test}.jsonl`.
The default launcher trains on `true_multihop` and evaluates on its dev split.

The extractive evaluator converts the predicted span back to text and reports
the same ROUGE-L, METEOR and optional BERTScore metrics as abstractive QA.

For a controlled comparison, the launcher passes the same encoder, data track,
span labels, optimizer, learning rate, batch/accumulation sizes, epoch or step
budget, passage loss weight, and evaluation metrics to every method. Each seed
uses the same reproducible shuffled sample order (`seed + epoch`) across methods;
the trainer also records SHA-256 hashes of all data splits, the manifest, and the
training/model/evaluation code. The resolved pretrained-encoder revision is
recorded and compared as well.
After all runs, `qa.summarize_extractive` verifies shared settings and data hashes
before writing `runs/extractive/true_multihop_comparison.json`. The default is a
single seed; use multiple paired seeds for a more robust comparison:

```bash
SEEDS="42 43 44" ./run_extractive.sh
```

Use a new `OUTPUT_ROOT` when changing settings; the audit rejects mixed or
stale completed runs.

Dev is evaluated by default. Once hyperparameters are fixed, evaluate the held-
out test split with `EVALUATE_TEST=1`; BERTScore can be enabled for all methods
with `SKIP_BERTSCORE=0`.

These implementations are controlled, shared-XLM-R, paper-inspired extractive
heads, not exact reproductions of the papers' full systems. In particular, the
original CoG/DFGN/HGN systems include entity/sentence-level reasoning and other
components not represented by this passage-level adaptation, while original
QANet uses a different encoder stack. Interpret results as a comparison of the
current adaptations under identical conditions, not as a reproduction of the
papers' published rankings.

## Information-retrieval experiments

Prepare an article-level retrieval corpus from all structured legal files, then
run the IR benchmark with one command:

```bash
./run_ir.sh
```

The generated collection is shared by every method: it contains the numbered
articles in `dataset/IR/structured_data`, plus the exact full-document
fallbacks needed by the existing qrels. `true_multihop` retains only queries
with at least two distinct relevant passage IDs. The split-specific query and
qrels files remain disjoint; dev is the default evaluation split and test is
only run with `EVALUATE_TEST=1` after choices are fixed.

The default method list includes SQLite FTS5 BM25 (using the four rarest
non-stopword query terms for efficient candidate matching), a fine-tuned
shared-encoder dense baseline, reciprocal-rank-fusion BM25+dense hybrid, and adaptations of
all six IR papers in `papers/IR`: MDR, M3, Baleen, MoPo, GMR and IRCoT. The
paper adaptations are deliberately explicit about what this dataset can
support:

- MDR uses evidence-conditioned dense retrieval with greedy hops.
- M3 combines contrastive retrieval with a binary relevance objective and
  single-/multi-hop rank fusion. The corpus has no FEVER claim-class labels, so
  this is not its original NLI objective.
- Baleen uses focused token-level MaxSim training/reranking and a deterministic
  extractive fact condenser; it does not reproduce the learned two-stage
  condenser or latent hop-ordering training.
- MoPo uses an EMA posterior encoder and KL regularization, with a
  query-focused extractive gold-evidence summary as posterior supervision.
- GMR fine-tunes mT5 to generate a constrained sequence of valid corpus
  passage IDs, not the full passage text used in the paper.
- IRCoT interleaves BM25 with greedy zero-shot reasoning from a configurable
  local sequence-to-sequence model (`IRCOT_REASONER`). It does not use the
  paper's hand-written few-shot CoT demonstrations.

Example controls:

```bash
METHODS="bm25 dense hybrid mdr m3 baleen mopo" MAX_STEPS=100 ./run_ir.sh
SEEDS="42 43 44" EVALUATE_TEST=1 ./run_ir.sh
METHODS="ircot" IRCOT_REASONER="google/flan-t5-small" ./run_ir.sh
```

Dense-family experiments share the encoder, data track, optimizer settings,
seed, paired query order, optimizer-update budget, and train/dev/test corpus and
qrels. Multi-hop methods rotate the supervised evidence hop between epochs, so
each dense-family method sees the same number of query examples and updates.
All runs use the same maximum
retrieval depth and report Recall@k, MRR@k, nDCG@k, MAP@k, and
complete-evidence recall. The comparison audit verifies query/qrels/corpus
hashes before writing `runs/ir/true_multihop_comparison.{json,tsv}`. GMR and
IRCoT involve a generative model and therefore have different model families
and inference costs; they should be reported as system-level comparisons, not
as compute-matched encoder ablations. GMR may return fewer than the shared
ranking depth because its output is a generated evidence sequence; the mean
returned candidates is recorded alongside its retrieval metrics.

QA answer metrics are also available as a separate downstream evaluation,
using the same single QA reader and greedy decoder for every retriever. They
are not substituted for ranking metrics. For example:

```bash
EVALUATE_QA=1 QA_CHECKPOINT="runs/qa/true_multihop_pathfid_seed42" \
  QA_METHOD=pathfid QA_TOP_K=10 ./run_ir.sh
```

This writes per-retriever ROUGE-L and METEOR (plus optional BERTScore with
`SKIP_BERTSCORE=0`) under each run directory, then audits the fixed reader and
summarizes the results in `runs/ir/true_multihop_qa_comparison.{json,tsv}`.
The reader sees the same top-ranked context count and per-passage token budget
for every retriever; by default, the 2,048-token input budget is divided across
up to ten retrieved passages.

Neural model downloads and full-corpus vector indexing happen when the
experiment is launched. `runs/ir/models/*/checkpoints` allow interrupted dense
and generative training to resume. Use a new `OUTPUT_ROOT` when changing
settings, and keep `EVALUATE_TEST=0` until model selection is complete.
