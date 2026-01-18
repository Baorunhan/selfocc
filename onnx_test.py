import torch
import torch.nn as nn

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # 假设输入64x64
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16
        x = self.pool(x)  # 8x8
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 导出模型
model = ComplexNet()
model.eval()

dummy_input = torch.randn(1, 3, 64, 64)

# 导出并验证
torch.onnx.export(
    model,
    dummy_input,
    "complex_model.onnx",
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# 验证导出的ONNX模型
import onnx
onnx_model = onnx.load("complex_model.onnx")
onnx.checker.check_model(onnx_model)
print("模型验证成功！")
print(f"输入: {onnx_model.graph.input[0].name}")
print(f"输出: {onnx_model.graph.output[0].name}")
