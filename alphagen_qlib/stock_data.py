from typing import List, Union, Optional, Tuple
from enum import IntEnum
from pathlib import Path
import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


_DEFAULT_QLIB_DATA_PATH = "qlib_data"
_QLIB_INITIALIZED = False


def initialize_qlib(qlib_data_path: str = _DEFAULT_QLIB_DATA_PATH) -> None:
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=qlib_data_path, region=REG_CN)
    global _QLIB_INITIALIZED
    _QLIB_INITIALIZED = True


def _load_calendar(data_path: str, freq: str) -> pd.DatetimeIndex:
    cal_path = Path(data_path) / "calendars" / f"{freq}.txt"
    dates = pd.read_csv(cal_path, header=None).iloc[:, 0]
    return pd.DatetimeIndex(pd.to_datetime(dates))


def _load_instrument_codes(data_path: str) -> List[str]:
    inst_path = Path(data_path) / "instruments" / "all.txt"
    df = pd.read_csv(inst_path, sep="\t", header=None)
    return sorted(df[0].str.strip().unique().tolist())


def _read_bin_feature(feature_path: Path, n_calendar: int) -> np.ndarray:
    if not feature_path.exists():
        return np.full(n_calendar, np.nan, dtype=np.float32)
    raw = np.fromfile(feature_path, dtype="<f")
    data = np.full(n_calendar, np.nan, dtype=np.float32)
    if len(raw) <= 1:
        return data
    start_idx = int(raw[0])
    values = raw[1:]
    end_idx = min(start_idx + len(values), n_calendar)
    if start_idx >= 0 and start_idx < n_calendar:
        n_copy = end_idx - start_idx
        data[start_idx:end_idx] = values[:n_copy]
    return data


class StockData:
    _qlib_initialized: bool = False

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None,
        freq: str = "day"
    ) -> None:
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._freq = freq
        data_tup = preloaded_data if preloaded_data is not None else self._get_data()
        self.data, self._dates, self._stock_ids = data_tup

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        data_path = _DEFAULT_QLIB_DATA_PATH
        cal = _load_calendar(data_path, self._freq)

        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        start_idx = cal.searchsorted(start_ts)
        end_idx = cal.searchsorted(end_ts)
        if end_idx < len(cal) and cal[end_idx] != end_ts:
            end_idx -= 1

        expanded_start = max(0, start_idx - self.max_backtrack_days)
        expanded_end = min(len(cal), end_idx + self.max_future_days + 1)

        all_codes = _load_instrument_codes(data_path)
        feature_names = [f.name.lower() for f in self._features]

        features_dir = Path(data_path) / "features"
        n_cal = len(cal)
        n_codes = len(all_codes)
        n_feat = len(feature_names)
        n_steps = expanded_end - expanded_start

        values = np.full((n_steps, n_feat, n_codes), np.nan, dtype=np.float32)
        for ci, code in enumerate(all_codes):
            code_dir = features_dir / code.lower()
            for fi, fname in enumerate(feature_names):
                fpath = code_dir / f"{fname}.{self._freq}.bin"
                full_data = _read_bin_feature(fpath, n_cal)
                values[:, fi, ci] = full_data[expanded_start:expanded_end]

        return (
            torch.tensor(values, dtype=torch.float, device=self.device),
            cal[expanded_start:expanded_end],
            pd.Index(all_codes)
        )

    def __getitem__(self, slc: slice) -> "StockData":
        "Get a subview of the data given a date slice or an index slice."
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (stop if stop is not None else self.n_days) + self.max_future_days + self.max_backtrack_days
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = data.isnan().reshape(-1, data.shape[-1]).all(dim=0).logical_not().nonzero().flatten()
        data = data[:, :, remaining]
        dt_fmt = "%Y-%m-%d %H:%M:%S" if self._freq != "day" else "%Y-%m-%d"
        return StockData(
            instrument=self._instrument,
            start_time=self._dates[start + self.max_backtrack_days].strftime(dt_fmt),
            end_time=self._dates[stop - 1 - + self.max_future_days].strftime(dt_fmt),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            freq=self._freq,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()])
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)  # type: ignore
        if exclusive and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"Date {date} is out of range: available [{self._start_time}, {self._end_time}]")
        return idx
    
    def find_date_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
        """
        Find a slice of indices corresponding to the given date range.
        For the input, both ends are inclusive. The output is a normal left-closed right-open slice.
        """
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time, exclusive=False)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
