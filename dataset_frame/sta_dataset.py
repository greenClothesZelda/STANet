import torch
import torch.utils.data as data
from pathlib import Path
import json
import pandas as pd


class STADataset(data.Dataset):
    def __init__(
        self,
        file_name,
        time_step=8,
        lag_window=None,
        root=None,
        use_weather=False,
        weather_file="meteorological_data.csv",
        weather_encoding="cp949",
        weather_target_columns=None,
    ):
        self.time_step = int(time_step)
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}.")
        if lag_window is None:
            lag_window = self.time_step
        self.lag_window = int(lag_window)
        if self.lag_window <= 0:
            raise ValueError(f"lag_window must be positive, got {self.lag_window}.")
        root = Path(root) if root is not None else Path("./data/raw")
        data_path = root / file_name
        if weather_target_columns is None:
            weather_target_columns = ["강수량(mm)", "기온(°C)", "습도(%)", "적설(cm)"]
        self.use_weather = use_weather
        self.weather_dim = 0
        self.weather_features = None
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
        self.near_demands = []
        self.od = []
        days = []
        times = []
        holidays = []
        for series in data["x"]:
            demand = series["demand"]
            near_demand = series.get("near_demands")
            if near_demand is None:
                near_demand = [0] * len(demand)
            if len(near_demand) != len(demand):
                raise ValueError(
                    "near_demands length must match demand length for each timestamp."
                )
            od = series.get("OD", [])
            day = series["day"]
            time = series["time"]
            holiday = series["holiday"]
            self.demands.append(demand)
            self.near_demands.append(near_demand)
            self.od.append(od)
            days.append(day_to_idx[day])
            times.append(time)
            holidays.append(1 if holiday else 0)
        #print(f'od: {self.od}')
        self.demands = torch.tensor(self.demands, dtype=torch.long)
        self.near_demands = torch.tensor(self.near_demands, dtype=torch.long)
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
        # Legacy aliases kept for backward compatibility.
        self.temporal_features["day_of_week"] = self.temporal_features["dow"]
        self.temporal_features["hour_of_day"] = self.temporal_features["hod"]
        self.temporal_features["is_holiday"] = self.temporal_features["holiday"]
        if self.use_weather:
            weather_path = root / weather_file
            weather_df = pd.read_csv(weather_path, encoding=weather_encoding)
            missing_cols = [
                col for col in weather_target_columns if col not in weather_df.columns
            ]
            if missing_cols:
                raise KeyError(
                    f"Missing weather columns in {weather_path}: {missing_cols}"
                )
            weather_data = weather_df[list(weather_target_columns)].fillna(0.0)
            weather_tensor = torch.tensor(
                weather_data.to_numpy(), dtype=torch.float32
            )
            demand_timesteps = self.demands.size(0)
            if weather_tensor.size(0) < demand_timesteps:
                pad_len = demand_timesteps - weather_tensor.size(0)
                pad = torch.zeros((pad_len, weather_tensor.size(1)), dtype=weather_tensor.dtype)
                weather_tensor = torch.cat([weather_tensor, pad], dim=0)
            elif weather_tensor.size(0) > demand_timesteps:
                weather_tensor = weather_tensor[:demand_timesteps]

            weather_mean = weather_tensor.mean(dim=0, keepdim=True)
            weather_std = weather_tensor.std(dim=0, keepdim=True)
            weather_tensor = (weather_tensor - weather_mean) / (weather_std + 1e-6)
            self.weather_features = weather_tensor
            self.weather_dim = weather_tensor.size(1)

        self.delta_t_last = torch.zeros_like(self.demands, dtype=torch.float)
        if self.demands.size(0) > 0:
            self.delta_t_last[0] = torch.where(
                self.demands[0] > 0,
                torch.zeros_like(self.delta_t_last[0]),
                torch.ones_like(self.delta_t_last[0]),
            )
        for i in range(1, self.demands.size(0)):
            self.delta_t_last[i] = torch.where(
                self.demands[i] > 0,
                torch.zeros_like(self.delta_t_last[i - 1]),
                self.delta_t_last[i - 1] + 1,
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

    def __len__(self):
        return len(self.demands) - self.time_step

    def __getitem__(self, idx):
        delta_t_last = self.delta_t_last[idx:idx + self.time_step].transpose(0, 1)  # (N, T)
        demand_series = self.demands[idx:idx + self.time_step].transpose(0, 1)  # (N, T)
        near_demand_series = self.near_demands[idx:idx + self.time_step].transpose(0, 1)  # (N, T)
        valid_mask = torch.ones_like(demand_series)

        lag_offsets = torch.arange(self.lag_window - 1, -1, -1)
        time_index = torch.arange(idx, idx + self.time_step).unsqueeze(1) - lag_offsets.unsqueeze(0)  # (T, L)
        valid_lag_mask = (time_index >= 0).long()  # (T, L)
        time_index = time_index.clamp(min=0)
        lag_values = self.demands[time_index]  # (T, L, N)
        lag_values = lag_values * valid_lag_mask.unsqueeze(-1)
        y_lag = lag_values.permute(2, 0, 1).contiguous()  # (N, T, L)
        near_lag_values = self.near_demands[time_index]  # (T, L, N)
        near_lag_values = near_lag_values * valid_lag_mask.unsqueeze(-1)
        near_y_lag = near_lag_values.permute(2, 0, 1).contiguous()  # (N, T, L)
        m_lag = valid_lag_mask.unsqueeze(0).expand(self.num_nodes, -1, -1).contiguous()  # (N, T, L)
        dow = self.temporal_features["dow"][idx:idx + self.time_step]
        hod = self.temporal_features["hod"][idx:idx + self.time_step]
        holiday = self.temporal_features["holiday"][idx:idx + self.time_step]
        weather = None
        if self.weather_features is not None:
            weather = self.weather_features[idx:idx + self.time_step]  # (T, C_w)

        demand_features = {
            "delta_t_last": delta_t_last,
            "y_lag": y_lag,
            "m_lag": m_lag,
            "near_y_lag": near_y_lag,
            "deactivation_period": delta_t_last,
            "demand_series": demand_series,
            "near_demand_series": near_demand_series,
            "valid_mask": valid_mask,
        }
        temporal_features = {
            "dow": dow,
            "hod": hod,
            "holiday": holiday,
            "day_of_week": dow,
            "hour_of_day": hod,
            "is_holiday": holiday,
        }
        if weather is not None:
            temporal_features["weather"] = weather

        return {
            "demand_features": demand_features,
            "temporal_features": temporal_features,
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
