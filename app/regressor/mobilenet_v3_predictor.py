import time
import cv2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

SCALE_MAPPING = {
    0: 1.6,
    1: 2.5,
    2: 4.0
}

class MultiTaskMobileNet(nn.Module):

    def __init__(self, num_scale_classes=3):
        super().__init__()

        backbone = models.mobilenet_v3_small(weights=None)

        self.features = backbone.features

        self.pressure_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, num_scale_classes)
        )

    def forward(self, x):

        x = self.features(x)

        pressure = self.pressure_head(x)
        scale = self.scale_head(x)

        return pressure.squeeze(), scale


class MobileNetV3Predictor:

    def __init__(self, model_path, device="cpu"):

        self.device = torch.device(device)

        self.model = MultiTaskMobileNet(num_scale_classes=3)

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image_bgr):

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(image_rgb)

        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        start = time.perf_counter()

        pred_pressure, pred_scale = self.model(tensor)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        pressure = float(pred_pressure.item())

        scale_class = int(
            torch.argmax(pred_scale, dim=1).item()
        )

        scale = SCALE_MAPPING[scale_class]

        return (
            pressure,
            scale,
            scale_class,
            latency_ms
        )

    def visualize(
        self,
        image,
        pressure,
        scale
    ):
        vis = image.copy()

        cv2.putText(
            vis,
            f"Pressure: {pressure:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"Scale: {scale:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return vis
 