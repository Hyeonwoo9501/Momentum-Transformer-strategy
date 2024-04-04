import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

class FinancialTimeSeriesDataset(Dataset):
    """금융 시계열 데이터셋을 위한 Dataset 클래스"""

    def __init__(self, csv_file, scaling=True):
        """

        :param csv_file: 데이터셋이 있는 csv 파일의 경로.
        :param scaling: 데이터 정규화를 적용할지 여부
        """
        self.data_frame = pd.read_csv(csv_file)
        self.scaling = scaling

        if self.scaling:
            self.scaler = StandardScaler()
            self.data_frame = pd.DataFrame(self.scaler.fit_transform(self.data_frame), columns=self.data_frame.columns)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_frame.iloc[idx]
        sample = torch.tensor(sample.values, dtype=torch.float)
        return sample

def get_data_loader(csv_file, batch_size=32, scaling=True):
    """dataloader를 생성하는 함수"""
    dataset = FinancialTimeSeriesDataset(csv_file=csv_file, scaling=scaling)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# 예제 사용법
# csv_file_path = 'path/to/your/data.csv'
# loader = get_data_loader(csv_file_path, batch_size=32, scaling=True)
# for batch in loader:
#     # 모델 학습 또는 평가에 batch 사용