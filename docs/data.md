# Data

The training scripts expect processed daily cross-section pickle files in:

```text
data/opensource/
  csi300_dl_train.pkl
  csi300_dl_test.pkl
  csi800_dl_train.pkl
  csi800_dl_test.pkl
```

Only the train and test files are required by the current scripts. Validation
files may be kept locally, but they are not used by the manuscript protocol.

## Expected Batch Layout

Each data loader yields one trading-day cross-section:

```text
N x T x F
```

where:

```text
N: number of stocks in the daily universe
T: lookback window length, 8
F: 222 total columns
   158 stock-level Alpha158 factors
    63 market-state features
     1 future-return label
```

The model input uses the first 221 columns. The label is read from the final
time step and final column.

## Processing Notes

The manuscript follows the MASTER-style processed data protocol:

```text
train period: first quarter of 2008 to first quarter of 2020
test period: third quarter of 2020 to fourth quarter of 2022
buffer: second quarter of 2020
prediction horizon: 5 trading days
lookback window: 8 trading days
```

During training, the code drops the top and bottom 2.5 percent of labels within
each daily cross-section and applies cross-sectional z-score normalization. At
evaluation, missing labels are dropped for metrics.

Large pickle files are excluded from Git by `.gitignore`. If you publish data,
use a release asset, object storage, or a data repository rather than committing
the files to the source tree.
