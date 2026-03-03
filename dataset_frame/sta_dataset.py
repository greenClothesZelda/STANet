import torch
import torch.utils.data as data
from pathlib import Path
import json
import warnings

import pandas as pd


class STADataset(data.Dataset):
    def __init__(self, file_name, time_step=8, root=None):
        self.time_step = time_step
        self.target_column = ['강수량(mm)', '기온(°C)', '습도(%)', '적설(cm)']
        root = Path(root) if root is not None else Path("./data/raw")
        data_path = root / file_name
        with open(data_path, "r") as f:
            data = json.load(f)
        self.num_nodes = len(data["nodes"])
        self.coordinates = [[] for _ in range(self.num_nodes)]
        self.areas = [node["size"] for node in data["nodes"]]

        # land_use composition
        land_keys = set()
        poi_keys = set()
        for node in data["nodes"]:
            land_use = node["composition"].get("land_use", {}) if isinstance(
                node["composition"], dict) else {}
            poi = node["composition"].get("poi", {}) if isinstance(
                node["composition"], dict) else {}
            land_keys.update(land_use.keys())
            poi_keys.update(poi.keys())
        land_keys = list(land_keys)
        poi_keys = list(poi_keys)

        self.land_key_to_idx = {key: idx for idx, key in enumerate(land_keys)}
        self.poi_key_to_idx = {key: idx for idx, key in enumerate(poi_keys)}

        self.composition = [
            [0 for _ in range(len(land_keys))] for _ in range(self.num_nodes)]
        self.poi = [[0 for _ in range(len(poi_keys))]
                    for _ in range(self.num_nodes)]

        for node in data["nodes"]:
            lat = node["lat"]
            lon = node["lon"]
            nid = node["node_id"]
            self.coordinates[nid] = [lat, lon]
            land_use = node["composition"].get("land_use", {}) if isinstance(
                node["composition"], dict) else {}
            poi = node["composition"].get("poi", {}) if isinstance(
                node["composition"], dict) else {}
            for comp, val in land_use.items():
                if comp in self.land_key_to_idx:
                    self.composition[nid][self.land_key_to_idx[comp]] = val
            for comp, val in poi.items():
                if comp in self.poi_key_to_idx:
                    self.poi[nid][self.poi_key_to_idx[comp]] = val

        day_to_idx = {"Mon": 0, "Tue": 1, "Wed": 2,
                      "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        self.demands = []
        self.od = []
        days = []
        times = []
        holidays = []
        for series in data["x"]:
            demand = series["demand"]
            od = series.get("OD", [])
            day = series["day"]
            time = series["time"]
            holiday = series["holiday"]
            self.demands.append(demand)
            self.od.append(od)
            days.append(day_to_idx[day])
            times.append(time)
            holidays.append(1 if holiday else 0)
        #print(f'od: {self.od}')
        self.demands = torch.tensor(self.demands, dtype=torch.long)
        self.max_demand = int(self.demands.max().item()) if self.demands.numel() > 0 else 0
        self.od_matrix = torch.zeros((len(self.demands), self.num_nodes, self.num_nodes), dtype=torch.float)
        for i, od_list in enumerate(self.od):
            for od in od_list:
                u, v, cnt = od["u"], od["v"], od["cnt"]
                self.od_matrix[i, u, v] = cnt
        self.temporal_features = {
            "dow": torch.tensor(days, dtype=torch.long),
            "hod": torch.tensor(times, dtype=torch.long),
            "holiday": torch.tensor(holidays, dtype=torch.long),
        }
        self.weather_features = self._load_weather_features(root)
        # Legacy aliases kept for backward compatibility.
        self.temporal_features["day_of_week"] = self.temporal_features["dow"]
        self.temporal_features["hour_of_day"] = self.temporal_features["hod"]
        self.temporal_features["is_holiday"] = self.temporal_features["holiday"]

        self.delta_t_last = torch.zeros_like(self.demands, dtype=torch.float)
        for i in range(1, self.demands.size(0)):
            self.delta_t_last[i] = torch.where(
                self.demands[i - 1] == 0,
                self.delta_t_last[i - 1] + 1,
                torch.zeros_like(self.delta_t_last[i - 1]),
            )
        self.deactivation_period = self.delta_t_last

        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float)
        self.composition = torch.tensor(self.composition, dtype=torch.float)
        self.areas = torch.tensor(self.areas, dtype=torch.float)

        eps = 1e-6
        mean_coords = torch.mean(self.coordinates, dim=0)
        std_coords = torch.std(self.coordinates, dim=0)
        self.coordinates = (self.coordinates -
                            mean_coords) / (std_coords + eps)
        if self.composition.numel() > 0:
            max_composition, _ = torch.max(self.composition, dim=0)
            self.composition = self.composition / (max_composition + eps)
        self.poi = torch.tensor(self.poi, dtype=torch.float)
        if self.poi.numel() > 0:
            max_poi, _ = torch.max(self.poi, dim=0)
            self.poi = self.poi / (max_poi + eps)
        mean_area = torch.mean(self.areas)
        std_area = torch.std(self.areas)
        self.areas = (self.areas - mean_area) / (std_area + eps)

    def _load_weather_features(self, root: Path) -> torch.Tensor:
        weather_path = root / "meteorological_data.csv"
        seq_len = len(self.demands)
        num_cols = len(self.target_column)
        weather_tensor = torch.zeros((seq_len, num_cols), dtype=torch.float)
        if not weather_path.exists():
            warnings.warn(f"Weather file not found: {weather_path}. Using zeros.")
            return weather_tensor

        try:
            weather_df = pd.read_csv(weather_path, encoding="cp949")
        except UnicodeDecodeError:
            weather_df = pd.read_csv(weather_path, encoding="utf-8")

        weather_df = weather_df.reindex(columns=self.target_column, fill_value=0.0)
        weather_df = weather_df[self.target_column].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        weather_values = weather_df.to_numpy(dtype="float32")

        copy_len = min(seq_len, weather_values.shape[0])
        if copy_len > 0:
            weather_tensor[:copy_len] = torch.from_numpy(weather_values[:copy_len])
        if weather_values.shape[0] != seq_len:
            if weather_values.shape[0] > seq_len:
                action = f"Truncating weather rows to first {copy_len}."
            else:
                action = f"Using {copy_len} rows and zero-padding remaining {seq_len - copy_len} rows."
            warnings.warn(
                f"Weather row count ({weather_values.shape[0]}) != demand length ({seq_len}). "
                f"{action}"
            )
        return weather_tensor

    def __len__(self):
        return len(self.demands) - self.time_step

    def __getitem__(self, idx):
        delta_t_last = self.delta_t_last[idx + self.time_step - 1].unsqueeze(1)
        y_lag = self.demands[idx:idx + self.time_step].transpose(0, 1)
        m_lag = torch.ones_like(y_lag)
        dow = self.temporal_features["dow"][idx:idx + self.time_step]
        hod = self.temporal_features["hod"][idx:idx + self.time_step]
        holiday = self.temporal_features["holiday"][idx:idx + self.time_step]
        weather_features = self.weather_features[idx:idx + self.time_step]

        return {
            "demand_features": {
                "delta_t_last": delta_t_last,
                "y_lag": y_lag,
                "m_lag": m_lag,
                "deactivation_period": delta_t_last,
                "demand_series": y_lag,
                "valid_mask": m_lag,
            },
            "temporal_features": {
                "dow": dow,
                "hod": hod,
                "holiday": holiday,
                "weather_features": weather_features,
                "meteo_features": weather_features,
                "day_of_week": dow,
                "hour_of_day": hod,
                "is_holiday": holiday,
            },
            "OD_matrix": self.od_matrix[idx:idx + self.time_step],  # (T, N, N),
            "labels": self.demands[idx + self.time_step],  # (N,)
        }


def stad_collate_fn(batch):
    """Collate function for Hugging Face Trainer batches."""
    demand_features = {
        key: torch.stack([sample["demand_features"][key]
                         for sample in batch], dim=0)
        for key in batch[0]["demand_features"]
    }
    temporal_features = {
        key: torch.stack([sample["temporal_features"][key]
                         for sample in batch], dim=0)
        for key in batch[0]["temporal_features"]
    }
    od_matrix = torch.stack([sample["OD_matrix"] for sample in batch], dim=0)
    labels = torch.stack([sample["labels"] for sample in batch], dim=0)
    return {
        "demand_features": demand_features,
        "temporal_features": temporal_features,
        "OD_matrix": od_matrix,
        "od_matrix": od_matrix,
        "labels": labels,
    }
