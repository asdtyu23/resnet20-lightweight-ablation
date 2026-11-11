import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from thop import profile
from models import ResNet20

# 自动下载数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 使用 dataset 文件夹，且 download=False 禁止它偷偷联网！
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def run_experiment(variant_name, use_dw, use_se, epochs=10):
    print(f"\n{'='*40}")
    print(f"开始实验: {variant_name}")
    print(f"{'='*40}")
    
    model = ResNet20(use_dw=use_dw, use_se=use_se).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 打印参数量和计算量
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"模型参数量: {params / 1e6:.4f} M")
    print(f"模型计算量: {flops / 1e6:.4f} M FLOPs")
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # 测试准确率
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | 测试准确率: {acc:.2f}%")

# ================= 运行入口 =================
if __name__ == '__main__':
    EPOCHS = 10
    run_experiment("Baseline (标准卷积)", use_dw=False, use_se=False, epochs=EPOCHS)
    run_experiment("DW (深度可分离)", use_dw=True, use_se=False, epochs=EPOCHS)
    run_experiment("DW + SE (注意力)", use_dw=True, use_se=True, epochs=EPOCHS)