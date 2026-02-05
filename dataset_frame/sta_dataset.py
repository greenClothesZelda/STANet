import torch
import torch.utils.data as data
from pathlib import Path
import json


class STADataset(data.Dataset):
    def __init__(self, file_name, time_step=8, root=None):
        self.time_step = time_step
        root = Path(root) if root is not None else Path('./data/raw')
        data_path = root / file_name
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.num_nodes = len(data['nodes'])
        self.coordinates = [[] for _ in range(self.num_nodes)]
        self.areas = [node['size'] for node in data['nodes']]

        composition_keys = set()
        for node in data['nodes']:
            for comp in node['composition']:
                composition_keys.add(comp)
        composition_keys = list(composition_keys)

        self.comp_key_to_idx = {key: idx for idx,
                                key in enumerate(composition_keys)}
        self.comp_idx_to_key = {idx: key for idx,
                                key in enumerate(composition_keys)}

        self.composition = [
            [0 for _ in range(len(composition_keys))] for _ in range(len(data['nodes']))]

        for node in data['nodes']:
            lat = node['lat']
            lon = node['lon']
            self.coordinates[node['node_id']] = [lat, lon]
            for comp in node['composition']:
                comp_idx = self.comp_key_to_idx[comp]
                self.composition[node['node_id']
                                 ][comp_idx] = node['composition'][comp]

        #print(f'x keys: {data["x"][0].keys()}')
        day_to_idx = {'Mon': 0, 'Tue': 1, 'Wed': 2,
                      'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        self.demands = []
        self.temporal_features = []
        days = []
        times = []
        holidays = []
        for series in data['x']:
            demand = series['demand']
            day = series['day']
            time = series['time']
            holiday = series['holiday']
            self.demands.append(demand)
            days.append(day_to_idx[day])
            times.append(time)
            holidays.append(1 if holiday else 0)

        # tensor로 변환
        self.demands = torch.tensor(self.demands, dtype=torch.long)
        self.temporal_features = {
            'day_of_week': torch.tensor(days, dtype=torch.long),
            'hour_of_day': torch.tensor(times, dtype=torch.long),
            'is_holiday': torch.tensor(holidays, dtype=torch.long)
        }
        self.deactivation_period = torch.zeros_like(
            self.demands, dtype=torch.float)
        for i in range(1, self.demands.size(0)):
            self.deactivation_period[i] = torch.where(
                self.demands[i-1] == 0, self.deactivation_period[i-1] + 1, torch.zeros_like(self.deactivation_period[i-1]))
        # metadata
        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float)
        self.composition = torch.tensor(self.composition, dtype=torch.float)
        self.areas = torch.tensor(self.areas, dtype=torch.float)

        # 각종 정규화 나중에 삭제할수도?
        # 좌표
        eps = 1e-6
        mean_coords = torch.mean(self.coordinates, dim=0)
        std_coords = torch.std(self.coordinates, dim=0)
        self.coordinates = (self.coordinates -
                            mean_coords) / (std_coords + eps)
        # 구역
        max_composition, _ = torch.max(self.composition, dim=0)
        self.composition = self.composition / (max_composition + eps)
        # 면적
        mean_area = torch.mean(self.areas)
        std_area = torch.std(self.areas)
        self.areas = (self.areas - mean_area) / (std_area + eps)

    def __len__(self):
        return len(self.demands) - self.time_step

    def __getitem__(self, idx):
        return {
            'demand_features': {
                # (N, 1)
                    # use last timestep in window; avoid out-of-bounds at sequence end
                    'deactivation_period': self.deactivation_period[idx + self.time_step].unsqueeze(1),
                # (N, T)
                'demand_series': self.demands[idx:idx + self.time_step].transpose(0, 1),
                # (N, T)
                'valid_mask': torch.ones_like(self.demands[idx:idx + self.time_step].transpose(0, 1)),
            },
            'temporal_features': {
                # (T,)
                'day_of_week': self.temporal_features['day_of_week'][idx:idx + self.time_step],
                # (T,)
                'hour_of_day': self.temporal_features['hour_of_day'][idx:idx + self.time_step],
                # (T,)
                'is_holiday': self.temporal_features['is_holiday'][idx:idx + self.time_step],
            },
            'labels': self.demands[idx + self.time_step]  # (N,)
        }


def stad_collate_fn(batch):
    """Collate function for Hugging Face Trainer batches."""
    demand_features = {
        key: torch.stack([sample['demand_features'][key]
                         for sample in batch], dim=0)
        for key in batch[0]['demand_features']
    }
    temporal_features = {
        key: torch.stack([sample['temporal_features'][key]
                         for sample in batch], dim=0)
        for key in batch[0]['temporal_features']
    }
    labels = torch.stack([sample['labels'] for sample in batch], dim=0)
    return {
        'demand_features': demand_features,
        'temporal_features': temporal_features,
        'labels': labels,
    }
