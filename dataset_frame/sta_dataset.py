import torch
import torch.utils.data as data
from pathlib import Path
import json

class STADataset(data.Dataset):
    def __init__(self, file_name):
        root = Path('./data/raw')
        data_path = root / file_name
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.coordinates = [[] for _ in range(len(data['nodes']))]
        self.areas = [node['size'] for node in data['nodes']]
        
        composition_keys = set()
        for node in data['nodes']:
            for comp in node['composition']:
                composition_keys.add(comp)
        composition_keys = list(composition_keys)
        
        self.comp_key_to_idx = {key: idx for idx, key in enumerate(composition_keys)}
        self.comp_idx_to_key = {idx: key for idx, key in enumerate(composition_keys)}
        
        self.composition = [[0 for _ in range(len(composition_keys))] for _ in range(len(data['nodes']))]
        
        for node in data['nodes']:
            lat = node['lat']
            lon = node['lon']
            self.coordinates[node['node_id']] = [lat, lon]
            for comp in node['composition']:
                comp_idx = self.comp_key_to_idx[comp]
                self.composition[node['node_id']][comp_idx] = node['composition'][comp]
        
        print(f'x keys: {data["x"][0].keys()}')
        day_to_idx = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}
        self.demands = []
        self.temporal_features = []
        for series in data['x']:
            demand = series['demand']
            day = series['day']
            time = series['time']
            holiday = series['holiday']
            self.demands.append(demand)
            self.temporal_features.append([day_to_idx[day], time, holiday])
        
        #tensor로 변환
        self.demands = torch.tensor(self.demands, dtype=torch.long)
        self.temporal_features = torch.tensor(self.temporal_features, dtype=torch.long)
        #metadata
        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float)
        self.composition = torch.tensor(self.composition, dtype=torch.float)
        self.areas = torch.tensor(self.areas, dtype=torch.float)
        
        #각종 정규화 나중에 삭제할수도?
        #좌표 
        eps = 1e-6
        mean_coords = torch.mean(self.coordinates, dim=0)
        std_coords = torch.std(self.coordinates, dim=0)
        self.coordinates = (self.coordinates - mean_coords) / (std_coords + eps)
        #구역
        max_composition, _ = torch.max(self.composition, dim=0)
        self.composition = self.composition / (max_composition + eps)
        #면적
        mean_area = torch.mean(self.areas)
        std_area = torch.std(self.areas)
        self.areas = (self.areas - mean_area) / (std_area + eps)
        
    def __len__(self):
        return len(self.demands)
    
    def __getitem__(self, idx):
        return {
            'demand': self.demands[idx],
            'temporal_features': self.temporal_features[idx],
        }