import cv2
import torch
import numpy as np
import pickle
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1. CẤU HÌNH THÔNG SỐ (ĐÃ SỬA)
# ============================================================
PKL_PATH = 'label_mapping.pkl'
CHECKPOINT_PATH = 'lsvit_fullclass_best.pth'

#  Đường dẫn tới video bạn muốn dự đoán
VIDEO_PATH = '267475.mp4'


# ============================================================
# 2. KIẾN TRÚC MÔ HÌNH (GIỮ NGUYÊN)
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
# 3. NẠP NHÃN & MÔ HÌNH VÀO RAM/GPU (GIỮ NGUYÊN)
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

if device.type == 'cuda':
    model = model.half()
    print("⚡ Đã kích hoạt chế độ siêu tốc FP16!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# 4. HỆ THỐNG HIỂN THỊ TIẾNG VIỆT (UNICODE) (GIỮ NGUYÊN)
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


def put_text_unicode(frame_bgr, text, pos, font, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = pos
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4], fill=bg_color)
    except AttributeError:
        pass
    draw.text((x, y), text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
# 5. XỬ LÝ VIDEO & DỰ ĐOÁN (VIẾT MỚI HOÀN TOÀN)
# ============================================================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video tại: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Đang xử lý video: {video_path} (Tổng số khung hình: {total_frames})")

    if total_frames < 16:
        print("⚠️ Cảnh báo: Video quá ngắn (< 16 frames). Kết quả có thể không chính xác.")
        # Nếu video quá ngắn, lấy tất cả frame hiện có và lặp lại frame cuối cho đủ 16
        frame_indices = np.clip(np.linspace(0, total_frames - 1, 16, dtype=int), 0, total_frames - 1)
    else:
        # Lấy mẫu đều 16 frames từ toàn bộ video
        frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)

    frames_to_process = []
    display_frames = []  # Lưu lại ảnh gốc để tí nữa chiếu lên xem
    current_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in frame_indices:
            # Lưu lại frame gốc để hiển thị
            display_frames.append(frame.copy())

            # Tiền xử lý để đưa vào model
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = transform(pil_img)
            frames_to_process.append(input_tensor)

        current_frame_idx += 1

    cap.release()

    # Xử lý trường hợp có frame bị lỗi không đọc được
    while len(frames_to_process) < 16:
        frames_to_process.append(frames_to_process[-1])
        display_frames.append(display_frames[-1])

    # Chuyển đổi list tensor thành shape (1, C, T, H, W)
    video_tensor = torch.stack(frames_to_process).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    if device.type == 'cuda':
        video_tensor = video_tensor.half()

    print("🧠 Đang phân tích cử chỉ...")

    # DỰ ĐOÁN
    with torch.no_grad():
        outputs = model(video_tensor)
        prob = torch.nn.functional.softmax(outputs.float(), dim=1)
        top_p, top_class = prob.topk(1, dim=1)

        confidence = top_p.item() * 100
        pred_label = idx_to_class[top_class.item()]

    result_text = f"Kết quả: {pred_label} ({confidence:.2f}%)"
    print(f"✅ {result_text}")

    # (Tùy chọn) Hiển thị lại video kèm theo kết quả dự đoán
    print("📺 Đang phát lại video với kết quả. Nhấn phím bất kỳ trên cửa sổ video để đóng.")
    for frame in display_frames:
        frame_with_text = put_text_unicode(frame, result_text, (20, 20), font_large, text_color=(0, 255, 255))
        cv2.imshow('Ket Qua Nhan Dien', frame_with_text)
        # Chờ 100ms mỗi frame để tạo hiệu ứng phát video (10fps)
        if cv2.waitKey(100) != -1:
            break

    cv2.waitKey(0)  # Dừng lại ở frame cuối cùng cho đến khi người dùng tắt
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_video(VIDEO_PATH)