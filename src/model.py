from models.transformer_encoder import TransformerWithOutputLayer

# model hyperparameter
input_dim = 20 # 입력 특성 수
d_model = 512 # 내부 특성 차원
num_heads = 8 # 멀티 헤드 어텐션의 헤드 수
num_layers = 6 # 인코더 레이어의 수
output_dim = 1 # 출력 차원 (시퀀스 위치마다 예측할 값의 수)
dropout_rate = 0.1 # 드롭아웃 비율

# 모델 인스턴스 생성
model = TransformerWithOutputLayer(input_dim, d_model, num_heads, num_layers, output_dim, dropout_rate)