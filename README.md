# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One step ahead forecasting of mobile Internet traffic in Milan, comparing three
recurrent architectures: a simple RNN, an LSTM and a GRU. The evaluation period is
the week of 16 to 22 December 2013, and the models are compared across the three
areas with the highest total Internet traffic.

Data is the Telecommunications SMS, Call, Internet activity dataset for Milan
(Barlacchi et al., 2015): 10,000 grid areas, readings every 10 minutes, 1 November
2013 to 1 January 2014.

## Headline results

Mean absolute error on the test week, averaged over five random seeds:

| Area | Persistence | Seasonal naive | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 5161 | 92.80 | 338.59 | 86.08 | 84.58 | **83.70** |
| 5059 | 81.52 | 171.74 | 73.96 | **67.57** | 69.12 |
| 5259 | 75.97 | 470.32 | 66.67 | **66.00** | 67.33 |

Three findings worth stating up front:

- **LSTM and GRU are not distinguishable.** Across fifteen paired runs the difference
  between them changes sign, so neither can be called better.
- **The baseline matters more than the architecture.** Against seasonal naive the
  models look 2.4 to 7 times better, but against persistence, which is the honest
  baseline when traffic correlates 0.987 with itself ten minutes earlier, the gain is
  only 10 to 17 percent.
- **One seed is not enough.** A single run made the GRU look like the winner in two
  of three areas. Five seeds reversed that.

## Repository layout

```
src/
  config.py           paths, constants, the evaluation window
  download_data.py    guestbook aware downloader, reduces each day to parquet
  consolidate.py      builds the compact files used by the analysis and Colab
  dataset.py          loading, chronological splits, scaling, sliding windows
  eda.py              distribution, focus areas, autocorrelation, decomposition
  spatial.py          the 100 x 100 grid, Moran's I, correlation against distance
  memory_profile.py   measured peak memory for six ways of loading a day
  reduction_chain.py  where the 20.8 GB to 357 MB reduction comes from
  models.py           the three recurrent models and two baselines
  train.py            shared training loop, early stopping, timing
  tune.py             staged hyperparameter search with a full experiment log
  diagnostics.py      error by hour and traffic level, bias, the echo test
  experiment.py       runs the whole study and writes every artefact
notebooks/
  milan_traffic.ipynb         the study as run on Colab with a GPU
results/                      generated tables and figures
```

## Setup

Python 3.10. Local work covers the download, reduction, memory profiling and
exploratory analysis. Model training was run on Google Colab with a Tesla T4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` leaves out PyTorch on purpose, because Colab already provides it
and the last release with x86 macOS wheels is 2.2.2. To run the model code locally:

```bash
pip install -r requirements-dev.txt
```

## Reproducing the results

### 1. Download and reduce the data

The dataset sits behind a Dataverse guestbook, so each download posts contact details
and receives a temporary signed link. Supply your own details through the environment.

```bash
export DATAVERSE_EMAIL="you@example.com"
export DATAVERSE_NAME="Your Name"
export DATAVERSE_INSTITUTION="Your Institution"

python src/download_data.py --workers 4
```

This fetches 62 daily files, 20.8 GB of tab separated text, and reduces each to about
27 MB of parquet. Around 2.5 to 3 hours; measured throughput plateaus near 2.1 MB/s
whatever the worker count. The run is resumable: finished days are recorded in
`logs/download_state.json` and skipped, and interrupted transfers continue with HTTP
range requests.

Keep one day uncompressed for the memory profiling step:

```bash
python src/download_data.py --limit 1 --keep-raw
```

### 2. Build the consolidated files

```bash
python src/consolidate.py
```

Writes `data/processed/colab/`: an 8928 x 10000 float32 matrix of Internet activity
(357 MB), its index, per area totals and the five areas of interest. This folder is
what gets uploaded to Google Drive.

### 3. Analysis and evidence

```bash
python src/eda.py
python src/spatial.py
python src/memory_profile.py
python src/reduction_chain.py
```

### 4. Forecasting experiments

On Colab, open `notebooks/milan_traffic.ipynb`, set the runtime to GPU, point
`DRIVE_DATA` at the uploaded folder and run in order. The full study including the
five seed repeats takes about fifteen minutes on a T4.

Locally, with PyTorch installed:

```bash
python src/experiment.py            # full staged search then final models
python src/experiment.py --quick    # short run to check the pipeline works
```

## Data handling

### Storage, 20.8 GB to 358 MB

`src/reduction_chain.py` regenerates this from the download log and the files on disk.

| stage | files | size | vs raw |
|---|---|---|---|
| Raw text as published | 62 | 20,804.8 MB | 1.0x |
| Daily parquet | 62 | 1,654.4 MB | 12.6x |
| Internet matrix | 1 | 357.1 MB | 58.3x |
| Upload folder | 4 | 357.9 MB | 58.1x |

Nothing is discarded. The final size is arithmetic: 8,928 time steps x 10,000 areas
is 89,280,000 values, and at four bytes each that is 357.1 MB, which is the file size
to within a 128 byte header. The saving comes from four kinds of overhead, not from
any loss of precision: numbers written as text, one row per country code, four
activity columns this study does not use, and a square id and timestamp repeated on
every one of 89 million rows that the matrix replaces with position.

### Memory during loading

Each approach runs in its own process and reports the peak the operating system
recorded, because the peak while loading is what has to fit in RAM.

| approach | peak | vs naive |
|---|---|---|
| Read it all, default types | 627.3 MB | 1.0x |
| Smaller number types | 352.0 MB | 1.8x |
| Only the columns we need | 138.9 MB | 4.5x |
| Chunked read and add up | 402.9 MB | 1.6x |
| Read the reduced parquet | 64.1 MB | 9.8x |
| One area from the matrix | 1.0 MB | 627x |

Loading all 62 days the naive way would peak near 39 GB against 16 GB of RAM, so the
reduction is what makes the study possible rather than merely faster.

### Limitations

The matrix is dense, so the 34,682 slots (0.039%) with no reading are stored as zero
and cannot be told apart from a genuine zero. That does not arise for the three busy
areas studied here, but it would for a quiet one. The matrix also keeps Internet only,
which is why the gzipped raw text is retained locally rather than deleted.

## Method notes

- Splits are chronological: train to 8 December, validate 9 to 15 December, test 16
  to 22 December. The test week is fixed by the assignment and never used for
  selection.
- Scaling statistics come from the training period only.
- Validation and test windows may reach back into the days before them, which is
  legitimate for one step ahead forecasting because those values have genuinely been
  observed by the time the prediction is made.
- The hyperparameter search runs in stages rather than as one grid, so each round is
  justified by the previous one. Every run is logged, including those that did not
  improve.
- Hyperparameters are tuned on the busiest area and reused across all three, so a
  difference between areas reflects the data rather than different settings.
- Final models are trained on five seeds and compared in pairs, because run to run
  variation is around 2 MAE and larger than most differences between models.
- Timings are taken with `time.perf_counter` after a device warm up and a CUDA
  synchronisation. Inference is the median of repeated passes.

## References

[1] G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi,
F. Antonelli, A. Vespignani, A. Pentland and B. Lepri, "A multi-source dataset of
urban life in the city of Milan and the Province of Trentino," *Scientific Data*,
vol. 2, 150055, 2015.

[2] Harvard Dataverse, Telecommunications SMS, Call, Internet MI.
doi:10.7910/DVN/EGZHFV

## Side experiment: a convolutional model

The main study compares three recurrent cells. `src/cnn_test.py` asks a separate
question: does a dilated causal convolution, which reads the whole window at once
rather than one step at a time, do any better on the same task?

```bash
python src/cnn_test.py --data-dir data/processed/colab
python src/cnn_test.py --data-dir /content/data --quick   # only checks it runs
```

Nothing in the main pipeline changes. The convolution presents the same interface as
an RNN layer, so it reuses the existing model wrapper, training loop, scaling and
metrics, which keeps the comparison fair for the same reason the recurrent comparison
is fair: only the cell differs. Depth is chosen automatically so the receptive field
covers the whole input window.

This is an add-on rather than a core contribution, and it is not part of the reported
results.
