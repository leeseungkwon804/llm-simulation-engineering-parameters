# Engineering parameters in large language model social simulation

Data and code for two manuscripts on how the implementation layer of a generative
agent-based simulation determines what the simulation reports.

- **Paper A.** *Engineering Parameters Decide What a Large Language Model Social Simulation Finds*
- **Paper B.** *Large Language Model Agents Answer on a Round Number Grid That Does Not Respond to Uncertainty*

Author: Sewon Min, Sahmyook University.

## What is here

Every call made during the study is released, including the calls that returned nothing.
No row has been removed.

| File | Rows | Contents |
|---|---|---|
| `data/exp2_haiku.jsonl` | 1500 | principal experiment, three arms, Haiku 5.5 |
| `data/exp3_sonnet.jsonl` | 810 | replication, Sonnet 5.5 |
| `data/exp3_opus.jsonl` | 810 | replication, Opus 5.5 |
| `data/exp3_fable.jsonl` | 810 | replication, Fable 5.1 |
| `data/cost.jsonl` | 360 | output tokens consumed per call at a budget of 1024 |
| `data/exp4.jsonl` | 720 | response-format and uncertainty manipulations |
| `data/main_haiku.jsonl` | 1360 | an earlier run, retained for completeness |
| `data/pilot.jsonl`, `data/pilot2.jsonl` | 44 | pilots, including the calls lost to the temperature error |

Each row carries the arm, model, persona level, task statement, agent identifier, raw reply,
parsed value, stop reason and token counts, and separates interface errors from non-responses.

## Code

| Script | Purpose |
|---|---|
| `code/harness.py` | prompt materials and the API client |
| `code/exp2.py` | principal experiment |
| `code/exp3.py` | replication on a second, third and fourth model |
| `code/exp4.py` | response-format and uncertainty manipulations |
| `code/cost.py` | output token consumption measurement |
| `code/verify.py` | re-derives every numeric claim in both manuscripts from the data |
| `code/tables.py` | generates the manuscript tables |
| `code/figs_p1.py`, `code/figs_rest.py` | generate the figures |

## Reproducing

```bash
export ANTHROPIC_API_KEY=...        # only needed to collect new data
python3 code/verify.py              # re-derives every number from the released files
python3 code/tables.py              # regenerates the tables
```

`verify.py` prints one line per claim. It is the script that caught three errors in a draft of the
manuscripts before submission.

## Note on the released calls

Responses were produced by language models acting as simulated participants. They are not human
subject data. No human or animal participants were involved and no ethics approval was required.

## Licence

MIT. See `LICENSE`.
