import torch
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']      # 支持中文
plt.rcParams['axes.unicode_minus'] = False        # 解决负号 '-' 显示为方块的问题

class_names = ['玻璃瓶', '菜叶菜帮', '茶叶渣', '厨房废品', '大骨头', '刀片', '瓜果皮壳', '肉类', '剩菜剩饭', '鱼骨鱼刺']

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

net = torchvision.models.resnet50(weights=None)
net.fc = nn.Linear(net.fc.in_features, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load('./models/test_best_10epoch_resnet50.pth'))
net.to(device)
net.eval()  # 评估模式
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join('datasets', folder),
    transform=transform_test) for folder in ['valid', 'test']]

batch_size = 32
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
# 第二部分，测试结果
all_preds = []
all_labels = []
top1_correct = 0
top5_correct = 0
total_samples = 0

with torch.no_grad():
    for X, y in tqdm(valid_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        top1_pred = y_hat.argmax(dim=1)
        top1_correct += (top1_pred == y).sum().item()

        top5_preds = y_hat.topk(5, dim=1).indices
        top5_correct += sum([y[i] in top5_preds[i] for i in range(len(y))])

        all_preds.extend(top1_pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        total_samples += y.size(0)

# ------------------------ Top-K Accuracy ------------------------
top1_acc = top1_correct / total_samples
top5_acc = top5_correct / total_samples
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")

# ------------------------ 分类报告 ------------------------
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_excel("classification_report.xlsx", index=True)  # 输出 Excel 文件
print("分类报告已保存为 classification_report.xlsx")

# ------------------------ 混淆矩阵 ------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()