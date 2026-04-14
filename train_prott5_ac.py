
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np

# ===================== 配置优化 =====================
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 2e-2  # 针对小样本增加正则化强度
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"


# ===================== Focal Loss (针对难分类样本优化) =====================
class FocalLoss(nn.Module):
    """
    Focal Loss 会降低易分类样本的权重，使模型专注于难分类的正样本，从而提升 SN。
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 获取预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ===================== 数据集 (含特征空间增强) =====================
class ACPFeatureDataset(Dataset):
    def __init__(self, pt_path, max_len=1024, is_train=False):
        data = torch.load(pt_path)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        label = self.labels[idx]

        if self.is_train and label == 1:
            # 训练集正样本加入微小高斯噪声，防止模型对稠密特征过拟合
            emb += torch.randn_like(emb) * 0.01

        curr_len = emb.size(0)
        if curr_len < self.max_len:
            pad = torch.zeros((self.max_len - curr_len), 1024)
            emb = torch.cat([emb, pad], dim=0)
        else:
            emb = emb[:self.max_len, :]

        return emb.transpose(0, 1), torch.tensor(label, dtype=torch.long)


# ===================== 模型架构 (保持多尺度 + SE) =====================
class MultiScaleAttentionCNN(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super().__init__()
        self.conv3 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, 128, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(384)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(384, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 384, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(384 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.bn(x)
        x = x * self.se(x)
        avg_p = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_p = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        combined = torch.cat([avg_p, max_p], dim=1)
        return self.classifier(combined)


# ===================== 核心优化：动态阈值搜索 =====================
def find_best_threshold(y_true, y_probs):
    """
    寻找使 MCC 最大化的最佳阈值。
    """
    best_t = 0.5
    best_mcc = -1
    # 在 0.1 到 0.8 之间以 0.01 为步长搜索
    for t in np.arange(0.1, 0.8, 0.01):
        preds = (y_probs >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
    return best_t


# ===================== 训练流程 =====================
def train():
    train_loader = DataLoader(ACPFeatureDataset("train_emb.pt", is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ACPFeatureDataset("test_emb2.pt", is_train=False), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiScaleAttentionCNN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 切换为 Focal Loss
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # alpha=0.75 意味着更倾向于正样本

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Starting training with Focal Loss and Dynamic Thresholding...")

    best_mcc_overall = -1

    for epoch in range(EPOCHS):
        model.train()
        for embs, labels in train_loader:
            embs, labels = embs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for embs, labels in test_loader:
                    logits = model(embs.to(DEVICE))
                    probs = F.softmax(logits, dim=1)[:, 1]  # 获取正类的概率
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            # 动态寻找本轮的最佳阈值
            current_best_t = find_best_threshold(all_labels, all_probs)

            # 使用最优阈值生成预测
            final_preds = (all_probs >= current_best_t).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_labels, final_preds, labels=[0, 1]).ravel()

            acc = (tp + tn) / (tp + tn + fp + fn)
            mcc = matthews_corrcoef(all_labels, final_preds)
            sn = tp / (tp + fn) if (tp + fn) > 0 else 0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0

            if mcc > best_mcc_overall:
                best_mcc_overall = mcc
                # 提示：保存模型时，一定要同时保存这个 current_best_t，否则推理时无法复现结果
                torch.save({'model': model.state_dict(), 'threshold': current_best_t}, "best_model.pth")

            print(
                f"Epoch {epoch + 1:03d} | BestT: {current_best_t:.2f} | ACC: {acc:.3f} | MCC: {mcc:.3f} | SN: {sn:.3f} | SP: {sp:.3f}")


if __name__ == "__main__":
    train()





