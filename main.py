import os
import sys

import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
model.eval()


def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)


class VideoProcessor(QThread):
    update_label = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def detect_and_save_people(self, frame, image_path):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform_image(image).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        with torch.no_grad():
            predictions = model(image_tensor)

        draw = ImageDraw.Draw(image)

        pred_scores = predictions[0]["scores"].cpu().numpy()
        pred_boxes = predictions[0]["boxes"].cpu().numpy()
        pred_labels = predictions[0]["labels"].cpu().numpy()

        people_count = 0
        for label, score, box in zip(pred_labels, pred_scores, pred_boxes):
            if label == 1 and score > 0.8:
                draw.rectangle(
                    [(box[0], box[1]), (box[2], box[3])], outline="red", width=3
                )
                people_count += 1

        # Создаем папку для сохранения, если она не существует
        output_folder = "detected_people"
        os.makedirs(output_folder, exist_ok=True)
        full_image_path = os.path.join(output_folder, image_path)

        if people_count > 0:
            image.save(full_image_path)

        return people_count

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        screenshot_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(10 * fps) == 0:
                screenshot_count += 1
                image_path = f"screenshot_{screenshot_count}.png"
                people_count = self.detect_and_save_people(frame, image_path)

                # Обновляем отчет с полным путем к сохраненному изображению
                full_image_path = os.path.join("detected_people", image_path)
                self.update_label.emit(
                    f"Скриншот #{screenshot_count} - Кол-во людей: {people_count} - Путь: {full_image_path}"
                )

            frame_count += 1

        cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video People Detector")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.button = QPushButton("Загрузить видео")
        self.button.clicked.connect(self.load_video)

        main_layout.addWidget(self.button)
        main_layout.addWidget(self.scroll_area)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)

        self.setCentralWidget(main_widget)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi)"
        )
        if video_path:

            self.video_processor = VideoProcessor(video_path)
            self.video_processor.update_label.connect(self.update_label)
            self.video_processor.start()

    def update_label(self, text):
        label = QLabel(text)
        self.scroll_layout.addWidget(label)


if __name__ == "__main__":  # Исправлено условие на корректное
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
