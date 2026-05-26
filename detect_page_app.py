import os
import sys

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

from detect_page_ui import Ui_DetectPage
from pca_axis_utils import compute_major_axis_comparison
from skeleton_endpoint_utils import compute_skeleton_farthest_endpoints
from usb_camera import UsbCamera

os.environ["YOLO_VERBOSE"] = "False"


class DetectPageWindow(QtWidgets.QMainWindow):
    SEGMENT_TARGETS = ("chicken",)
    PCA_AXIS_ORIGINAL_COLOR = (0, 255, 255)
    PCA_AXIS_OPTIMIZED_COLOR = (255, 255, 0)
    PCA_ENDPOINT_REGION_RATIO = 0.15
    # ENDPOINT_METHOD = "skeleton"
    ENDPOINT_METHOD = "pca"
    CONF_THRESHOLD = 0.5

    def __init__(self):
        super().__init__()
        self.ui = Ui_DetectPage()
        self.ui.setupUi(self)

        self.model = None
        self.current_path = ""
        self.camera = UsbCamera()
        self.latest_camera_frame = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._read_camera_frame)

        self.ui.btnOpen.clicked.connect(self.open_source)
        self.ui.btnCapture.clicked.connect(self.capture_and_detect)
        self.ui.btnStop.clicked.connect(self.stop_camera)
        self.ui.radioImage.toggled.connect(self._on_mode_changed)
        self.ui.radioCamera.toggled.connect(self._on_mode_changed)
        self.ui.comboModel.currentIndexChanged.connect(self.load_model)
        self.ui.comboCamera.currentIndexChanged.connect(self._on_camera_index_changed)

        self._set_placeholder(self.ui.leftPanel, "左侧原图 / 摄像头预览")
        self._set_placeholder(self.ui.rightPanel, "右侧检测结果")
        self.update_mode_label()
        self.load_model()
        self._sync_controls()

    def load_model(self):
        path = self.ui.comboModel.currentText().strip()
        if not path:
            return

        if os.path.exists(path):
            try:
                self.model = YOLO(path)
                self.ui.statusBar.showMessage(f"模型加载完成: {path}", 3000)
                return
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "模型加载失败", str(exc))
                return

        QtWidgets.QMessageBox.critical(
            self,
            "模型未找到",
            "没有找到可加载的 YOLO 模型文件，请检查 rknn/ 目录。",
        )
        raise FileNotFoundError("YOLO model not found")

    def _panel_label(self, panel):
        return panel.layout().itemAt(0).widget()

    def _set_placeholder(self, panel, text):
        label = self._panel_label(panel)
        label.setText(text + "\n等待加载")
        label.setPixmap(QtGui.QPixmap())

    def _sync_controls(self):
        camera_mode = self.ui.radioCamera.isChecked()
        self.ui.comboCamera.setEnabled(camera_mode)
        self.ui.btnCapture.setEnabled(camera_mode)

    def update_mode_label(self):
        mode = "图片" if self.ui.radioImage.isChecked() else "摄像头"
        method = self.ENDPOINT_METHOD.upper()
        self.ui.labelMode.setText(f"当前模式: {mode} | 端点方法: {method}")

    def _on_mode_changed(self):
        self.update_mode_label()
        self._sync_controls()
        if self.ui.radioCamera.isChecked():
            self.open_source()
        else:
            self.stop_camera()
            self.current_path = ""
            self.ui.labelPath.setText("未选择文件")
            self._set_placeholder(self.ui.leftPanel, "左侧原图 / 摄像头预览")

    def _on_camera_index_changed(self):
        if self.ui.radioCamera.isChecked():
            self.open_camera()

    def open_source(self):
        if self.ui.radioImage.isChecked():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if path:
                self.load_image(path)
            return

        self.open_camera()

    def load_image(self, path):
        self.stop_camera()
        img = cv2.imread(path)
        if img is None:
            self.ui.statusBar.showMessage("图片读取失败", 3000)
            return

        self.current_path = path
        self.ui.labelPath.setText(path)
        self._show_frame(img, self._panel_label(self.ui.leftPanel))

        result_img, segment_text = self._infer(img)
        self._show_frame(result_img, self._panel_label(self.ui.rightPanel))
        self.ui.textCoords.setPlainText(segment_text)
        self.ui.statusBar.showMessage("图片检测完成", 3000)

    def open_camera(self):
        camera_index = self.ui.comboCamera.currentData()
        self.stop_camera(clear_result=False)

        if not self.camera.open(camera_index):
            self.ui.statusBar.showMessage(f"摄像头 {camera_index} 打开失败", 5000)
            self._set_placeholder(self.ui.leftPanel, "左侧原图 / 摄像头预览")
            return

        self.current_path = f"USB 摄像头 {camera_index}"
        self.ui.labelPath.setText(self.current_path)
        self.latest_camera_frame = None
        self.timer.start(33)
        self.ui.statusBar.showMessage(f"摄像头 {camera_index} 已连接，点击拍照检测", 5000)

    def _read_camera_frame(self):
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.stop_camera(clear_result=False)
            self.ui.statusBar.showMessage("摄像头读取失败，已停止预览", 5000)
            self._set_placeholder(self.ui.leftPanel, "左侧原图 / 摄像头预览")
            return

        self.latest_camera_frame = frame.copy()
        self._show_frame(frame, self._panel_label(self.ui.leftPanel))

    def capture_and_detect(self):
        if not self.ui.radioCamera.isChecked():
            self.ui.statusBar.showMessage("请先切换到摄像头模式", 3000)
            return

        if self.latest_camera_frame is None:
            self.ui.statusBar.showMessage("当前没有可用的摄像头画面", 3000)
            return

        captured = self.latest_camera_frame.copy()
        result_img, segment_text = self._infer(captured)
        self._show_frame(captured, self._panel_label(self.ui.leftPanel))
        self._show_frame(result_img, self._panel_label(self.ui.rightPanel))
        self.ui.textCoords.setPlainText(segment_text)
        self.ui.statusBar.showMessage("拍照检测完成", 3000)

    def _infer(self, frame):
        if self.model is None:
            return frame, "模型未加载"

        # results = self.model(frame)[0]
        results = self.model(frame, conf=self.CONF_THRESHOLD)[0]
        plotted = results.plot()
        segment_text = self._extract_target_segments(results)
        plotted = self._draw_target_points(plotted, results)
        plotted, endpoint_text = self._draw_target_axis(plotted, results)
        if endpoint_text:
            segment_text = (
                segment_text + "\n\n" + endpoint_text if segment_text else endpoint_text
            )
        return plotted, segment_text

    def _draw_target_points(self, image, results):
        if results.masks is None or results.boxes is None:
            return image

        polygons = getattr(results.masks, "xy", None)
        if polygons is None:
            return image

        classes = results.boxes.cls.tolist()
        names = results.names

        for idx, cls_id in enumerate(classes):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            if class_name not in self.SEGMENT_TARGETS:
                continue
            if idx >= len(polygons):
                continue

            points = np.asarray(polygons[idx], dtype=np.int32)
            if points.size == 0:
                continue

            for x, y in points:
                cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)

        return image

    def _draw_target_axis(self, image, results):
        if results.masks is None or results.boxes is None:
            return image, ""

        polygons = getattr(results.masks, "xy", None)
        if polygons is None:
            return image, ""

        classes = results.boxes.cls.tolist()
        names = results.names
        info_lines = []
        object_indices = {name: 1 for name in self.SEGMENT_TARGETS}

        for idx, cls_id in enumerate(classes):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            if class_name not in self.SEGMENT_TARGETS:
                continue
            if idx >= len(polygons):
                continue

            points = np.asarray(polygons[idx], dtype=np.float32)
            if points.shape[0] < 2:
                continue

            endpoint_result = self._compute_endpoints(points)
            if endpoint_result is None:
                continue

            self._draw_endpoint_result(image, endpoint_result)
            object_index = object_indices[class_name]
            info_lines.extend(
                self._format_endpoint_info(class_name, object_index, endpoint_result)
            )
            object_indices[class_name] += 1

        return image, "\n".join(info_lines).strip()

    def _compute_endpoints(self, points):
        method = self.ENDPOINT_METHOD.strip().lower()
        if method == "skeleton":
            return compute_skeleton_farthest_endpoints(points)
        if method == "pca":
            return compute_major_axis_comparison(
                points,
                endpoint_region_ratio=self.PCA_ENDPOINT_REGION_RATIO,
            )
        raise ValueError(f"Unsupported endpoint method: {self.ENDPOINT_METHOD}")

    def _draw_endpoint_result(self, image, endpoint_result):
        center_pt = endpoint_result["center_pt"]

        if endpoint_result.get("method") == "skeleton":
            for x, y in endpoint_result.get("skeleton_pixels", []):
                cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
            p1 = endpoint_result["p1"]
            p2 = endpoint_result["p2"]
            cv2.line(
                image,
                p1,
                p2,
                self.PCA_AXIS_OPTIMIZED_COLOR,
                3,
                cv2.LINE_AA,
            )
            cv2.circle(image, p1, 7, (255, 0, 0), -1)
            cv2.circle(image, p2, 7, (255, 128, 0), -1)
            cv2.circle(image, center_pt, 6, (0, 0, 255), -1)
            return

        original_p1 = endpoint_result["original_p1"]
        original_p2 = endpoint_result["original_p2"]
        optimized_p1 = endpoint_result["optimized_p1"]
        optimized_p2 = endpoint_result["optimized_p2"]
        cv2.line(
            image,
            original_p1,
            original_p2,
            self.PCA_AXIS_ORIGINAL_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            optimized_p1,
            optimized_p2,
            self.PCA_AXIS_OPTIMIZED_COLOR,
            3,
            cv2.LINE_AA,
        )
        cv2.circle(image, optimized_p1, 7, (255, 0, 0), -1)
        cv2.circle(image, optimized_p2, 7, (255, 128, 0), -1)
        cv2.circle(image, center_pt, 6, (0, 0, 255), -1)

    def _format_endpoint_info(self, class_name, object_index, endpoint_result):
        if endpoint_result.get("method") == "skeleton":
            return [
                f"{class_name} #{object_index} endpoint method: skeleton",
                f"{class_name} #{object_index} endpoint p1: {endpoint_result['p1']}",
                f"{class_name} #{object_index} endpoint p2: {endpoint_result['p2']}",
                f"{class_name} #{object_index} center: {endpoint_result['center_pt']}",
                f"{class_name} #{object_index} skeleton endpoints: {endpoint_result['endpoint_count']}",
                f"{class_name} #{object_index} path length: {endpoint_result['path_length']:.2f}",
            ]

        return [
            f"{class_name} #{object_index} endpoint method: pca",
            f"{class_name} #{object_index} original axis p1: {endpoint_result['original_p1']}",
            f"{class_name} #{object_index} original axis p2: {endpoint_result['original_p2']}",
            f"{class_name} #{object_index} optimized axis p1: {endpoint_result['optimized_p1']}",
            f"{class_name} #{object_index} optimized axis p2: {endpoint_result['optimized_p2']}",
            f"{class_name} #{object_index} center: {endpoint_result['center_pt']}",
        ]

    def _extract_target_segments(self, results):
        if results.masks is None or results.boxes is None:
            return "未检测到 chicken 分割结果"

        names = results.names
        lines = []
        classes = results.boxes.cls.tolist()
        polygons = getattr(results.masks, "xy", None)
        if polygons is None:
            return "当前结果没有可用的分割轮廓坐标"

        object_indices = {name: 1 for name in self.SEGMENT_TARGETS}
        for idx, cls_id in enumerate(classes):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            if class_name not in self.SEGMENT_TARGETS:
                continue
            if idx >= len(polygons):
                continue

            points = np.asarray(polygons[idx], dtype=np.float32)
            if points.size == 0:
                continue

            rounded_points = np.round(points).astype(int).tolist()
            object_index = object_indices[class_name]
            lines.append(f"{class_name} #{object_index}:")
            lines.append(str(rounded_points))
            lines.append("")
            object_indices[class_name] += 1

        if not lines:
            return "未检测到 chicken 分割结果"

        return "\n".join(lines).strip()

    def stop_camera(self, clear_result=True):
        if self.timer.isActive():
            self.timer.stop()
        self.camera.release()
        self.latest_camera_frame = None
        if clear_result:
            self.ui.textCoords.clear()

    def _show_frame(self, frame, label):
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)
        label.setText("")

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DetectPageWindow()
    win.show()
    sys.exit(app.exec())
