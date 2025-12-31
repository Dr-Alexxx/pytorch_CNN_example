import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import yaml
import os

# ----------------------------
# 1. 结构定义 (必须与 SimpleCNN 训练脚本完全一致)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate, classifier_dropout):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 2. 预测逻辑
# ----------------------------
def predict():
    # A. 加载配置
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # B. 实例化并加载模型权重
    # 从配置中读取参数确保结构完全相同
    model = SimpleCNN(
        num_classes=cfg['model']['num_classes'],
        dropout_rate=cfg['model']['dropout_rate'],
        classifier_dropout=cfg['model']['classifier_dropout']
    ).to(device)

    model_path = cfg['saving']['model_path']
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}，请先运行训练脚本。")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已加载: {model_path}")

    # C. 弹窗选择文件
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择 CIFAR-10 图片进行预测",
        filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("未选择图片。")
        return

    # D. 预处理 (与训练脚本 transform 保持一致)
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # 确保输入尺寸为 32x32
        transforms.ToTensor(),
        transforms.Normalize(cfg['normalization']['mean'], cfg['normalization']['std'])
    ])

    try:
        image = Image.open(file_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # E. 推理过程
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        # F. 展示结果
        label = classes[predicted.item()]
        score = confidence.item() * 100
        result_msg = f"预测类别: {label}\n置信度: {score:.2f}%"

        print(f"\n文件: {file_path}")
        print("-" * 20)
        print(result_msg)
        
        messagebox.showinfo("预测结果", result_msg)

    except Exception as e:
        messagebox.showerror("错误", f"处理图片时出错: {e}")

if __name__ == '__main__':
    predict()