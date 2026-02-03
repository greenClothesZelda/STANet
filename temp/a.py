from models import *
from dataset_frame import *
 
if __name__ == "__main__":
    dataset = STADataset('graph_data.json')
    print(f'dataset.coordinates: {dataset.coordinates}')
    print(f'dataset.composition: {dataset.composition}')
    for sample in dataset:
        print(sample)
        break