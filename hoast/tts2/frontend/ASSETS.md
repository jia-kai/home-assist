# Offline frontend assets

`Frontend(artifact_dir: Path, language: Literal["EN", "ZH"])` loads only
repository-owned algorithms and explicit local resources. Construction does
not read `$HOME/nltk_data`, download files, import Melo/FunASR/Transformers, or
run subprocesses. Preparation does not execute speech or BERT models.

## Optional explicit English-resource reuse

`Frontend(artifact_dir, language, english=existing_frontend.english)` reuses the
provided English dictionary, POS tagger, GRU arrays and bounded OOV cache. It
does not read `cmudict.rep` or `g2p/` from that constructor's artifact directory.
Its tokenizer, checkpoint-specific symbol IDs and Chinese resources are still
loaded separately; EN and ZH retain their independent language/tone behavior.
No global resource cache or automatic cross-checkpoint reuse is introduced.

Before reuse, the caller must verify identical manifest checksum records for
`cmudict.rep` and **all `g2p/` paths**, including the pronunciation/tagger assets.
The caller must serialize shared use, for example with the engine's TTS lock,
because the explicitly shared OOV cache is mutable. Without the keyword, each
frontend constructs independent English resources. This is a candidate RAM
optimization; memory/latency benefits require measurement against independent
resource objects rather than being assumed.

## Required files

| File                          | Content                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| `config.json`                 | Checkpoint config with an ordered `symbols` string list.       |
| `tokenizer.json`              | Complete fast BERT tokenizer, including normalizer and CLS/SEP.|
| `cmudict.rep`                 | Melo's syllabified CMU dictionary, UTF-8 with comment headers.  |
| `g2p/g2p.npz`                 | Unmodified `g2p_en` 2.1.0 `checkpoint20.npz`, renamed.           |
| `g2p/homographs.en`            | `g2p_en` pipe-separated spelling/pron1/pron2/POS-prefix table.  |
| `g2p/cmudict.json`             | JSON object: lowercase word → list of ARPAbet pronunciations.  |
| `g2p/tagger/weights.json`      | NLTK perceptron weights: feature → tag → numeric weight.        |
| `g2p/tagger/tagdict.json`      | NLTK perceptron unambiguous word → POS-tag dictionary.          |
| `g2p/tagger/classes.json`      | JSON list of NLTK perceptron POS-tag classes.                   |
| `opencpop-strict.txt` (ZH)     | Melo strict pinyin → space-separated phones, tab-delimited.     |

The checkpoint symbol list is authoritative; symbols must be unique, `_`
must have ID zero, and `UNK` must exist. Generated valid phonemes absent from
the checkpoint alphabet are errors, not implicit unknowns. Always retain
upstream licenses with copied resource files.

Use **bert-base-uncased** for EN and **bert-base-multilingual-uncased** for ZH.
Export `tokenizer.json` from the corresponding fast tokenizer, preserving
WordPiece continuation prefixes and the BERT normalizer. Truncation and
padding are disabled by the frontend. Exactly one `[CLS]` and one `[SEP]`
are required. Ensure a BERT feature graph can accept the resulting sequence
length; the frontend does not silently truncate long utterances.
The pinned official BERT JSON omits the optional WordPiece model `type` field;
preparation accepts that schema after verifying its explicit WordPiece decoder,
continuation prefix, unknown token and vocabulary. The tokenizer file is copied
unchanged and native `Tokenizer.from_file` handles both schemas.

The `g2p` JSON files can be produced from previously prepared local NLTK
resources: serialize `cmudict.dict()`, `tagger.model.weights`, `tagger.tagdict`
and `list(tagger.classes)`. To preserve reference homograph decisions,
export the reference tagger's trained values, rather than substituting a
newly trained model. The runtime constructs `PerceptronTagger(load=False)`
and populates those values directly. It does not unpickle assets or perform
NLTK resource lookup. Artifact preparation is a separate concern.

The NPZ contains float arrays with validated dimension relationships. Let H be
the common encoder/decoder hidden width and Eenc/Edec their embedding widths:

- `enc_emb`: `[29,Eenc]`; `dec_emb`: `[74,Edec]`.
- `enc_w_ih`: `[3H,Eenc]`; `dec_w_ih`: `[3H,Edec]`.
- `enc_w_hh`, `dec_w_hh`: `[3H,H]`.
- `enc_b_ih`, `enc_b_hh`, `dec_b_ih`, `dec_b_hh`: `[3H]`.
- `fc_w`: `[74,H]`; `fc_b`: `[74]`.

The cached distributed `g2p_en` checkpoint has **H=Eenc=Edec=256**. Upstream
source comments describe smaller shapes and are not the artifact specification.
The frontend validates actual parameter relationships without resizing weights.

## Native offline preparation

The graph preparer calls the standard-library-only helper **after** copying
checkpoint `config.json` and the matching `tokenizer.json` into its staging
root:

```python
from tools.prepare_tts2_frontend import populate_frontend_assets

emitted = populate_frontend_assets(
    source,
    staging,
    language,
    g2p_resource_dir=explicit_g2p_data_directory,
    nltk_resource_dir=explicit_unpacked_nltk_directory,
)
```

`emitted` contains all generated/copied paths relative to the staging root,
including licenses, notices, the CMU README and `frontend-assets.json`
provenance. **Inventory `staging.rglob("*")`, with relative POSIX paths**, not
only its immediate files: runtime resources are nested under `g2p/tagger/`.
The helper rejects existing emitted paths; use the graph preparer's temporary
staging directory and atomic publication rather than overwriting artifacts.

The explicit NLTK directory contains `corpora/cmudict/{cmudict,README}` and
`taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle`.
The reference pickle contains only a weights dictionary, a tag dictionary and
a class set; a restricted preparation-only reader permits no globals except
the built-in set constructor. Native Python 3.14 converts it directly to JSON
without importing NLTK or executing an alternate interpreter. Current unpacked
`averaged_perceptron_tagger_eng` JSON files are accepted when the reference
pickle is absent. ZIP-only resources must first be unpacked explicitly.

For an offline CLI with mandatory, self-describing source/resource/output
arguments, run `python -m tools.prepare_tts2_frontend --help`. No downloader or
implicit NLTK data-path discovery is provided.

## Runtime dependencies

`numpy`, `tokenizers`, `inflect`, `nltk`; ZH additionally uses `cn2an`,
`jieba`, and `pypinyin`. No `g2p_en` dependency is required. Suitable current
Python 3.14 versions are `tokenizers>=0.22`, `inflect>=7.5`, `nltk>=3.9.2`,
`cn2an>=0.5.23`, `jieba>=0.42.1`, and `pypinyin>=0.55`. The parent integration
owns dependency installation and version qualification.

## Exact output geometry

`prepare` returns `TextInputs` with int64 phone/tone/language arrays `[1,P]`,
BERT IDs `[1,S]`, post-blank repeat counts `[S]`, and normalized text.
`len(word2ph) == S` and `sum(word2ph) == P`. Pre-blank counts are doubled,
then the first count is incremented. Blank phone/tone/language IDs are zero.
Original phone positions carry EN language ID two or ZH/mixed ID three.
English stress is mapped to tones 1/2/3 and offset by seven; consonant and
punctuation tones in EN have offset seven. Chinese tones remain unshifted;
English segments in mixed text get the English offset. Boundary pad tones
are seven for EN and zero for ZH.

## Compatibility and limits

- CMU-rep pronunciations take precedence, including homographs present in
  that dictionary. Fallback homographs receive upstream standalone-word POS
  tagging, not sentence-context retagging.
- English clock-time expansion intentionally corrects upstream content defects:
  whitespace before trailing words is preserved, dotted and undotted AM/PM are
  recognized, and midnight is twelve AM rather than PM. These changes prevent
  accidental word concatenation and incorrect spoken meridiems; other number
  and abbreviation conventions retain Melo fidelity.
- English OOV pronunciation uses the upstream float32 NumPy encoder/decoder,
  with the same twenty-step greedy limit. Empty predictions are errors.
- `[UNK]` tokenizer words use their source spelling for G2P instead of dropping
  unknown/OOV input or attempting to pronounce the literal `[UNK]` marker.
- Unsupported residual tokens receive an explicit `UNK` phone, never zero
  repeats merely because their pronunciation was unavailable.
- Mixed normalization follows Melo's restricted basic-Han/ASCII-English
  cleaner, including Chinese expansion of numerals inside English segments.
  Characters outside that cleaner's alphabet are removed intentionally.
- Character phones and English groups are aligned against the *full* artifact
  tokenizer encoding using offsets; whitespace has no pronunciation.
- The frontend does not support Japanese, Korean, French, Spanish, arbitrary
  scripts, contextual English homograph upgrades, or model execution.
- Jieba/pypinyin data versions can affect word segmentation and polyphonic
  readings. Qualify dependency versions against the chosen checkpoint/reference.
