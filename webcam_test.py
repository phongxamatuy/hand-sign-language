import cv2
import torch
import numpy as np
import pickle
import torch.nn as nn
import torchvision.models as models
from collections import deque
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import time

# ============================================================
# 1. CẤU HÌNH THÔNG SỐ
# ============================================================
PKL_PATH = 'label_mapping.pkl'
# file model tải trên gg drive
CHECKPOINT_PATH = 'model/lsvit_fullclass_best.pth'


FRAME_SKIP = 4
CONFIDENCE_THRESHOLD = 50.0


# ============================================================
# 2. KIẾN TRÚC MÔ HÌNH (BẢN ULTIMATE)
# ============================================================
class LSViT(nn.Module):
    def __init__(self, num_classes=100, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.3, num_frames=16):
        super(LSViT, self).__init__()
        self.num_frames = num_frames

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames + 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in list(self.projector.modules()) + list(self.classifier.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.feature_extractor(x).view(B * T, -1)
        features = self.projector(features).view(B, T, -1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        features = features + self.positional_encoding[:, :T + 1, :]
        out = self.transformer(features)
        return self.classifier(out[:, 0, :])


# ============================================================
# 3. NẠP NHÃN & MÔ HÌNH VÀO RAM/GPU
# ============================================================
with open(PKL_PATH, 'rb') as f:
    full_mapping = pickle.load(f)
all_classes = sorted(full_mapping.keys(), key=lambda x: full_mapping[x])
idx_to_class = {i: cls_name for i, cls_name in enumerate(all_classes)}
NUM_CLASSES = len(all_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang khởi động AI trên: {device.type.upper()}")

model = LSViT(num_classes=NUM_CLASSES, num_frames=16).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Bật Mixed Precision (FP16) cho GPU để tăng tốc x2
if device.type == 'cuda':
    model = model.half()
    print("⚡ Đã kích hoạt chế độ siêu tốc FP16!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# 4. HỆ THỐNG HIỂN THỊ TIẾNG VIỆT (UNICODE)
# ============================================================
def find_font(size=32):
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf"
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


font_large = find_font(36)
font_small = find_font(24)


def put_text_unicode(frame_bgr, text, pos, font, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = pos

    # Hỗ trợ cả Pillow bản mới (textbbox) và cũ (textsize)
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4], fill=bg_color)
    except AttributeError:
        pass  # Nếu dùng bản Pillow quá cũ thì bỏ qua vẽ viền đen

    draw.text((x, y), text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
# 5. VÒNG LẶP WEBCAM THỜI GIAN THỰC
# ============================================================
frame_buffer = deque(maxlen=16)
cap = cv2.VideoCapture(0)

# Giảm độ phân giải capture để CPU không bị nghẽn
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Không thể kết nối Webcam!")
    exit()

print("✅ Webcam sẵn sàng! Đưa tay lên và trải nghiệm. Nhấn 'q' để thoát.")

current_prediction = "Đang thu thập khung hình..."
frame_count = 0
fps_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # [LOGIC THỜI GIAN MỚI]: Cứ 3 frame thì lấy 1 frame đưa vào bộ nhớ
    if frame_count % FRAME_SKIP == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        input_tensor = transform(pil_img)
        frame_buffer.append(input_tensor)

        # Ngay khi buffer vừa nhận được tấm ảnh mới (và đã đủ 16 tấm), cho AI đoán luôn
        if len(frame_buffer) == 16:
            video_tensor = torch.stack(list(frame_buffer)).permute(1, 0, 2, 3).unsqueeze(0).to(device)

            # Đồng bộ kiểu dữ liệu (FP16) với mô hình
            if device.type == 'cuda':
                video_tensor = video_tensor.half()

            with torch.no_grad():
                outputs = model(video_tensor)
                prob = torch.nn.functional.softmax(outputs.float(), dim=1)
                top_p, top_class = prob.topk(1, dim=1)

                confidence = top_p.item() * 100
                pred_label = idx_to_class[top_class.item()]

                if confidence > CONFIDENCE_THRESHOLD:
                    current_prediction = f"{pred_label} ({confidence:.1f}%)"
                else:
                    current_prediction = "Đang xem xét..."

    # Tính toán FPS hiển thị
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_time)
        fps_time = time.time()

    # Hiển thị Chữ lên màn hình
    frame = put_text_unicode(frame, current_prediction, (15, 15), font_large, text_color=(0, 255, 0))
    frame = put_text_unicode(frame, f"FPS: {fps:.1f} | Buffer: {len(frame_buffer)}/16", (15, 70), font_small,
                             text_color=(255, 255, 0))

    cv2.imshow('AI Nhan Dien Ngon Ngu Ky Hieu - LSViT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()