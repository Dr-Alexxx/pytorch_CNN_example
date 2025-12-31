import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import yaml
import os

# ----------------------------
# 1. 结构定义 (必须与训练脚本中的模型结构一致)
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10, classifier_dropout=0.1):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ----------------------------
# 2. 预测主逻辑
# ----------------------------
def predict():
    # A. 加载配置
    with open('config_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # B. 加载模型
    model = OptimizedCNN(num_classes=cfg['model']['num_classes']).to(device)
    model_path = cfg['saving']['model_path']
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    # 加载权重并处理映射
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"成功加载模型: {model_path}")

    # C. 弹出文件选择对话框
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择一张图片进行预测",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("未选择任何文件，退出。")
        return

    # D. 图像预处理 (必须与训练时的测试集 transform 一致)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(cfg['normalization']['mean'], cfg['normalization']['std'])
    ])

    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # E. 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    # F. 输出结果
    result_text = f"预测结果: {classes[predicted.item()]} (置信度: {confidence.item()*100:.2f}%)"
    print("\n" + "="*30)
    print(f"图片路径: {file_path}")
    print(result_text)
    print("="*30)

    # 弹窗显示结果（可选）
    tk.messagebox.showinfo("预测完成", result_text)

if __name__ == "__main__":
    predict()