# Basic-punctuation routing evaluation

## Current input policy

Normalization version 2 retains basic punctuation: `,，.。!?！？;；:：'`.
Contractions, possessives, and initials remain readable: `don't`, `Adele's`, and
`G.E.M.`. Curly word apostrophes become ASCII apostrophes, while Chinese sentence
punctuation retains its width. Title brackets such as `《》`, decorative double
quotes, and other symbols are removed.

The same rule applies to STT output, model-input user turns, and generated data.
Structured schemas and tool outputs retain their syntax. Volume arguments remain
integers in `[0, 100]`: set requires a value; raise/lower default to five percentage
points unless an amount is supplied. Unsupported numeric forms are rejected, not
rewritten into different values.

## Rebuilt data and speech provenance

| Component                         | Rows / recordings |
| --------------------------------- | ----------------- |
| Canonical training                | 7,510             |
| Canonical validation              | 890               |
| Canonical test                    | 894               |
| Verified raw speech recordings    | 3,556             |
| Eligible training observations    | 3,818             |
| Added nonduplicate speech examples | 3,580             |
| Augmented training                | 11,090            |
| Held-out validation speech        | 744               |
| Held-out test speech              | 748               |

The existing recordings were reused, not presented as new TTS/STT executions.
Their raw transcripts were normalized again with punctuation retained. Exact
source text/split, audio checksums, model/voice/decoder settings, and a verified
inference implementation fingerprint were checked. Original inference provenance
is preserved in each migrated record.

Audio is shared by on/off contexts and equivalent clean/curated-error sources.
All held-out source IDs remain represented. Added training examples do not overlap
held-out canonical or speech inputs; identical model inputs have consistent labels.
The independent speech test contains all 748 observations in this run, with no
cross-split input collisions requiring quarantine.

## Retraining

- Base: `LiquidAI/LFM2.5-350M`, revision
  `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- LoRA rank 16, alpha 32, dropout 0.05, q/k/v projections; BF16, learning rate
  2e-4, seed 42, three epochs.
- Native tokenization and completion-only loss, without duplicate special tokens.
- Training used the 11,090-row corpus. Checkpoint selection used validation loss,
  not test results: **epoch 1 / step 174**, loss **0.0444334**.
- The adapter was reloaded and merged, then exported to INT8-weight OpenVINO.
  Both test sets were executed through the actual OpenVINO GenAI backend.

See [README.md](README.md#lora-training-and-int8-export) for portable commands,
artifact layout, and deployment instructions.

## Current-policy results

Exact ordered tool names and typed arguments are required. Omitted relative
volume amounts use the runtime's five-point default. Empty, malformed, or
truncated responses fail. No live tool handlers are executed by this evaluation.

| Model                         | Text test, 894 rows  | Speech test, 748 rows |
| ----------------------------- | ------------------- | -------------------- |
| Previous checkpoint, BF16      | 693 / 894 — 77.52%  | Not measured here    |
| Retrained checkpoint, BF16     | 692 / 894 — 77.40%  | 284 / 748 — 37.97%   |
| Retrained checkpoint, INT8     | **706 / 894 — 78.97%** | **284 / 748 — 37.97%** |

The previous checkpoint was evaluated on the **same version-2 text examples and
targets**. Retraining did not establish a text-routing improvement over that
checkpoint in BF16; their scores differ by one example. Precision and backend
differences prevent attributing the INT8/BF16 difference solely to quantization.

| Retrained INT8 speech slice | Correct / rows | Accuracy |
| -------------------------- | -------------- | -------- |
| Full intended-command test | 284 / 748      | 37.97%   |
| Audit-eligible transcripts  | 248 / 328      | 75.61%   |
| Review-required transcripts | 36 / 420       | 8.57%    |
| Volume                     | 104 / 128      | 81.25%   |
| Named music                | 80 / 500       | 16.00%   |

The eligible subset is selected by label-fidelity checks and is not an unbiased
substitute for the full speech result. Review-required examples retain intended
source targets for end-to-end accounting; many have lost or altered names and
cannot be treated as clean extraction examples. Abstention is 6/12 on both text
and speech tests, still a small and imperfect slice.

### Relation to the earlier letters-only policy

The preceding INT8 run scored 74.16% on its letters-only text set and 39.97% on
its speech set. The current run scores 78.97% and 37.97%, respectively. Input and
target spelling, retained training variants, and retraining all changed, so this
is not an isolated causal test of punctuation. In particular, preserving basic
punctuation **has not solved the full speech-routing problem**.

## Actual examples

Correct:

- `Choose Don't Stop Me Now as the song to play.`
  → `play_music(title="Don't Stop Me Now")`.
- `Put on G.E.M. 演唱的 Yellow，谢谢。`
  → `play_music(title='Yellow', artist='G.E.M.')`.
- `Describe the weather conditions in Paris, France tomorrow.`
  → `get_weather(period='tomorrow', city='Paris, France')`.
- `Raise the volume.` → `volume_music(action='louder', level=5)`.

Incorrect:

- `说说明天家里的天气情况。` → current weather instead of tomorrow.
- `Lower the volume.` → asks for an amount instead of using the five-point default.
- `My music selection is Adele's recording of Don't Stop Me Now.`
  → artist-only playback, dropping the specified title.
- A speech request for `Adele` was recognized with `Dale`; this remains flagged
  as a changed name rather than becoming an automatic training label.

## Limits

The recordings are synthetic and use a limited voice set; entity vocabularies are
shared between splits. On/off twins are related observations. Raw STT emitted
title brackets in 31 of the 3,556 recordings, illustrating that punctuation can
appear but is not dependable. Title brackets are removed under both policies.
Recognition of names and code-switching remains a major bottleneck, and routing
errors persist even with faithful transcripts. These results do not establish
readiness for autonomous deployment or performance on real microphone audio.
