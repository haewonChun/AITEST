import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 1. Dataset 정의
class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = f"{root_dir}/leftImg8bit/{split}"
        self.masks_dir = f"{root_dir}/gtFine/{split}"
        
        # 각 split당 500개 이미지만 사용
        self.images = list(torch.hub.list_dir_files(self.images_dir, "*_leftImg8bit.png"))[:500]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = f"{self.images_dir}/{img_name}"
        mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        mask_path = f"{self.masks_dir}/{mask_name}"
        
        image = torchvision.io.read_image(image_path)
        mask = torchvision.io.read_image(mask_path)[0]  # 단일 채널로 변환
        
        if self.transform:
            image = self.transform(image.float() / 255.0)
            mask = mask.long()
            
        return image, mask, img_name

# 2. 데이터 변환 정의
def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((128, 256)),  # 작은 이미지 크기
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 256)),  # 작은 이미지 크기
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

# 3. 경량화된 UNet++ 모델 정의
def create_model(num_classes):
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",        # 가벼운 백본
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        decoder_channels=(128, 64, 32, 16, 8),  # 감소된 디코더 채널
        decoder_attention_type=None      # 어텐션 제거
    )
    return model

# 4. Loss Function 정의
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# 5. IoU 계산 함수
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection / union))
            
    return torch.tensor(ious).nanmean().item()

# 6. 훈련 함수
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    
    for images, masks, _ in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 7. 체크포인트 저장 함수
def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f"{save_dir}/final_model.pth")

# 8. 추론 및 결과 저장 함수
def inference(model, test_loader, device, save_dir):
    model.eval()
    
    color_map = torch.tensor([
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [70, 70, 70],    # building
        [102, 102, 156], # wall
        [190, 153, 153], # fence
        [153, 153, 153], # pole
        [250, 170, 30],  # traffic light
        [220, 220, 0],   # traffic sign
        [107, 142, 35],  # vegetation
        [152, 251, 152], # terrain
        [70, 130, 180],  # sky
        [220, 20, 60],   # person
        [255, 0, 0],     # rider
        [0, 0, 142],     # car
        [0, 0, 70],      # truck
        [0, 60, 100],    # bus
        [0, 80, 100],    # train
        [0, 0, 230],     # motorcycle
        [119, 11, 32],   # bicycle
    ], device=device)

    with torch.no_grad():
        for images, _, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            for pred, img_name in zip(predictions, image_names):
                colored_pred = color_map[pred]
                torchvision.utils.save_image(
                    colored_pred.permute(2, 0, 1).float() / 255.0,
                    f"{save_dir}/pred_{img_name}"
                )

# 메인 훈련 루프
def main():
    # 기본 디렉토리 설정
    base_dir = '/content/drive/MyDrive/cityscapes_segmentation'
    checkpoints_dir = f"{base_dir}/checkpoints"
    predictions_dir = f"{base_dir}/predictions"
    
    # 하이퍼파라미터 설정
    num_classes = 19
    batch_size = 64      # 큰 배치 사이즈
    num_epochs = 10      # 적은 에폭 수
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 및 데이터로더 초기화
    train_dataset = CityscapesDataset(
        root_dir='/content/drive/MyDrive/cityscapes',
        split='train',
        transform=get_transform('train')
    )
    
    test_dataset = CityscapesDataset(
        root_dir='/content/drive/MyDrive/cityscapes',
        split='val',  # validation set 사용
        transform=get_transform('test')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델, 손실함수, 옵티마이저 초기화
    model = create_model(num_classes).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    print(f"Training started: {num_epochs} epochs")
    
    # 훈련 루프
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # 2 에폭마다 검증
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_ious = []
                for images, masks, _ in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    predictions = torch.argmax(outputs, dim=1)
                    val_ious.append(calculate_iou(predictions, masks, num_classes))
                
                val_iou = sum(val_ious) / len(val_ious)
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, IoU: {val_iou:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
    
    # 최종 모델 저장
    save_checkpoint(model, optimizer, num_epochs-1, train_loss, checkpoints_dir)
    
    # 추론 수행
    print("Running inference on test set...")
    inference(model, test_loader, device, predictions_dir)
    print("Training and inference completed!")

if __name__ == '__main__':
    # Google Drive 마운트
    drive.mount('/content/drive')
    
    # 필요한 디렉토리 생성
    base_dir = '/content/drive/MyDrive/cityscapes_segmentation'
    for dir_name in ['checkpoints', 'predictions']:
        dir_path = f"{base_dir}/{dir_name}"
        torch.hub.create_dir_if_not_exists(dir_path)
    
    main()
