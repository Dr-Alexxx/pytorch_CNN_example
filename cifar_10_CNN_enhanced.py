# cifar10_cnn_optimized.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp_torch  # 用于混合精度训练 - 使用新API
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

# 检查 torch.compile 可用性
def check_compile_availability():
    try:
        # 这里建议去掉 print，或者只返回状态，由 main 函数决定是否 print
        # 但为了最小改动，保持原样也可以，只要不放在全局即可
        compile_available = True
        print("Torch compile is available.")
    except ImportError:
        compile_available = False
        print("Torch compile is not available. Using standard execution.")
    return compile_available

# 注意：删除了这里的 compile_available = check_compile_availability()

def main():
    # 1. 移到这里！
    compile_available = check_compile_availability()

def main():
    # ----------------------------
    # 1. 从 YAML 文件加载配置
    # ----------------------------
    with open('config_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 训练参数
    batch_size = int(cfg['training']['batch_size'])  # 确保是 int
    epochs = int(cfg['training']['epochs'])  # 确保是 int
    learning_rate = float(cfg['training']['learning_rate'])  # 确保是 float
    weight_decay = float(cfg['training']['weight_decay'])  # <--- 添加 float() 转换

    # 数据集参数
    data_root = cfg['data']['root']
    num_workers = int(cfg['data']['num_workers'])  # 确保是 int
    download_data = cfg['data']['download']

    # 模型参数
    num_classes = int(cfg['model']['num_classes'])  # 确保是 int
    dropout_rate = float(cfg['model']['dropout_rate'])  # 确保是 float
    classifier_dropout = float(cfg['model']['classifier_dropout'])  # 确保是 float

    # 数据增强参数
    random_crop_padding = int(cfg['data_augmentation']['random_crop_padding'])  # 确保是 int
    random_hflip_prob = float(cfg['data_augmentation']['random_hflip_prob'])  # 确保是 float
    cutout_n_holes = int(cfg['data_augmentation']['cutout_n_holes'])  # 确保是 int
    cutout_length = int(cfg['data_augmentation']['cutout_length'])  # 确保是 int

    # 标准化参数
    normalize_mean = cfg['normalization']['mean']
    normalize_std = cfg['normalization']['std']

    # 保存路径
    model_save_path = cfg['saving']['model_path']
    plot_save_path = cfg['saving']['plot_path']

    print(f"Loading data from: {data_root}")
    print(f"Download flag is set to: {download_data}")

    # ----------------------------
    # 2. 数据预处理与加载 (使用增强策略)
    # ----------------------------
    # 训练集变换：包含多种增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=random_crop_padding, padding_mode='reflect'), # reflect padding often works better
        transforms.RandomHorizontalFlip(p=random_hflip_prob),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 如果 PyTorch 版本支持，可启用
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)), # 替代 Cutout
        # Cutout 可以用 RandomErasing 实现，或自定义
        # Cutout(n_holes=cutout_n_holes, length=cutout_length) # 需要自定义 Cutout 类
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download_data, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) # pin_memory=True 加速数据传输

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download_data, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) # pin_memory=True 加速数据传输

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ----------------------------
    # 3. 定义改进后的 CNN 模型 (ResNet-inspired blocks)
    # ----------------------------
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # Shortcut connection (identity or projection)
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
            out = self.relu(out)
            return out

    class OptimizedCNN(nn.Module):
        def __init__(self, num_classes=num_classes, dropout_rate=dropout_rate, classifier_dropout=classifier_dropout):
            super(OptimizedCNN, self).__init__()
            # Initial convolution
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Optional pooling for downsampling

            # Residual blocks
            self.layer1 = self._make_layer(32, 32, 2, stride=1) # 32 -> 32 channels
            self.layer2 = self._make_layer(32, 64, 2, stride=2) # 32 -> 64 channels, spatial downsample
            self.layer3 = self._make_layer(64, 128, 2, stride=2) # 64 -> 128 channels, spatial downsample
            # self.layer4 = self._make_layer(128, 256, 2, stride=2) # Optional deeper layer

            # Global Average Pooling (replaces Flatten + large Linear layer)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(classifier_dropout)
            self.fc = nn.Linear(128, num_classes) # Input size is 128 (last layer's out_channels)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            # x = self.maxpool(x) # Uncomment if using maxpool
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.layer4(x) # Uncomment if using deeper layer
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    # ----------------------------
    # 4. 初始化模型、损失函数和优化器
    # ----------------------------
    model = OptimizedCNN().to(device)

    # 尝试编译模型以加速 - 已禁用
    # if compile_available:
    #     model = torch.compile(model)
    #     print("Model compiled for speedup.")
    print("Torch compile is available but disabled due to missing Triton dependency on Windows. Proceeding without compilation.")
    print("Model loaded without torch.compile.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 使用 AdamW

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # 余弦退火调度

    # 混合精度训练 - 使用新API
    # 使用新的 torch.amp.GradScaler API
    scaler = amp_torch.GradScaler('cuda', enabled=True) # 使用新API

    # ----------------------------
    # 5. 训练函数 (集成混合精度)
    # ----------------------------
    def train_one_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # non_blocking=True for pin_memory

            optimizer.zero_grad()

            # 前向传播 + 混合精度
            # 使用新的 torch.amp.autocast API
            with amp_torch.autocast('cuda', enabled=True): # 使用新API
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 反向传播 + 混合精度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() # 更新缩放器

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        return avg_loss, acc

    # ----------------------------
    # 6. 测试函数 (集成混合精度)
    # ----------------------------
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # non_blocking=True

                # 前向传播 + 混合精度
                # 使用新的 torch.amp.autocast API
                with amp_torch.autocast('cuda', enabled=True): # 使用新API
                    outputs = model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        return acc

    # ----------------------------
    # 7. 主训练循环
    # ----------------------------
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_acc = 0.0
    print("Start training (optimized)...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch()
        test_acc = evaluate()
        scheduler.step() # 更新学习率

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch + 1}/{epochs}] - Best model saved with Test Acc: {test_acc:.2f}%")

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")

    print("Training finished (optimized)!")
    print(f"Best Test Accuracy achieved: {best_test_acc:.2f}%")

    # ----------------------------
    # 8. 可视化训练过程
    # ----------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Curve (Optimized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title('Accuracy Curve (Optimized)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

    # 保存最终模型 (可选)
    final_model_path = model_save_path.replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")


if __name__ == '__main__':
    main()