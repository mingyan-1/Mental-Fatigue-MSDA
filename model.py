import torch
import torch.nn as nn

class G_erp(nn.Module):
    def __init__(self):
        super(G_erp, self).__init__()

        # 卷积模块：多层卷积 + BatchNorm + ReLU
        self.conv_layers = nn.Sequential(
            nn.Conv1d(20, 32, kernel_size=3, padding=1),  # 增加通道数
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # 使用更大的卷积核提取多尺度特征
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # 双层 LSTM 层：捕捉更深的时序特征
        self.lstm = nn.LSTM(
            input_size=128,  # 输入特征维度
            hidden_size=64,  # 隐藏层大小
            num_layers=2,  # 双层 LSTM
            bidirectional=True,  # 双向 LSTM
            batch_first=True,  # batch_size 在第一维度
            dropout=0.5  # 增加 dropout 以减少过拟合
        )

        # 注意力机制：在 LSTM 输出后增强关键特征
        self.attention = nn.Sequential(
            nn.Linear(128, 64),  # 注意力权重
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 全连接层：深度 + 非线性激活
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 120, 256),  # 根据 LSTM 输出调整输入维度
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 加入 Dropout 减少过拟合
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8)  # 最终输出
        )
    def forward(self, input_data):
        # 输入 (batch_size, 20, 120)
        x = input_data.to(torch.float32).reshape(-1, 20, 120)
        x = self.conv_layers(x)  # 输出 (batch_size, 128, 120)
    
        # 将通道和序列维度互换，适应 LSTM 输入 (batch_size, 120, 128)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # 输出 (batch_size, 120, 128) 双层 LSTM
        # 使用注意力机制
        attn_weights = self.attention(x)  # 计算注意力权重
        x = x * attn_weights  # 广播机制应用权重
        x = x.reshape(x.size(0), -1)  # 将 x 展开为 (batch_size, 128 * 120)

        # 全连接层处理
        x = self.fc_layers(x)  # 输出 (batch_size, 8)
        return x

class G_ECG(nn.Module):

    def __init__(self):
        super(G_ECG, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('c_fc1', nn.Linear(10, 8))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(8))
        self.feature.add_module('c_relu1', nn.ReLU(True))
        # self.feature.add_module('f_pool1', nn.MaxPool1d(3, 2))


    def forward(self, input_data):
        input_data = input_data.reshape(-1, 10)
        input_data = input_data.to(torch.float32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 8)

        return feature



class G_EMG(nn.Module):

    def __init__(self):
        super(G_EMG, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(6, 8))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(8))
        self.feature.add_module('c_relu1', nn.ReLU(True))
        # self.feature.add_module('f_pool1', nn.MaxPool1d(3, 2))

    def forward(self, input_data):
        input_data = input_data.reshape(-1, 6)
        input_data = input_data.to(torch.float32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 8)

        return feature



class ResClassifier(nn.Module):
    def __init__(self, ch_in = 3, reduction=16):
        super(ResClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, ch_in),
            nn.Sigmoid()
        )
        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(True),
            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, 2),
    #        nn.Softmax()
        )


    def forward(self, input_data):
        model_flatten = nn.Flatten()
        flatten_data = model_flatten(input_data)
        b, c, _ = input_data.size()
        y = self.avg_pool(input_data).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1)  # FC获取通道注意力权重，是具有全局信息的
        class_output = self.class_classifier(input_data * y.expand_as(input_data))
        return class_output, flatten_data



