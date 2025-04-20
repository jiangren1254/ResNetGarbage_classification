import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
class_names = [
    "刀片",
    "剩菜剩饭",
    "厨房废品",
    "大骨头",
    "玻璃瓶",
    "瓜果皮壳",
    "肉类",
    "茶叶渣",
    "菜叶菜帮",
    "鱼骨鱼刺"
]

device = 'cpu'
# device = 'cuda'

net = torchvision.models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 10)  
net.load_state_dict(torch.load('./models/test_best_100epoch_resnet18.pth', map_location=device))
net = net.to(device)
net.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_path = './datasets/test/unknown/img_ (10).png' 
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device) 

with torch.no_grad():
    output = net(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()

print(f"预测类别索引：{class_names[pred_class]}")
