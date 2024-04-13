import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(device, data_loader, model, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in data_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')
    print("Training complete.")

###############
### example ###
###############
# def main():
#     # 모델 하이퍼파라미터 설정
#     input_dim = 10
#     d_model = 512
#     num_heads = 8
#     num_layers = 6
#     output_dim = 1
#     dropout_rate = 0.1
#     learning_rate = 0.001
#     num_epochs = 20
#
#     # 데이터 로드
#     dataset = FinancialTimeSeriesDataset('path/to/your/data.csv')
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     # 디바이스 설정
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 모델 초기화
#     model = TransformerWithOutputLayer(input_dim, d_model, num_heads, num_layers, output_dim, dropout_rate)
#     model.to(device)
#
#     # 손실 함수 및 옵티마이저 설정
#     criterion = nn.MSELoss()  # 회귀 문제라 가정
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # 모델 학습
#     train_model(data_loader, model, criterion, optimizer, num_epochs=num_epochs)