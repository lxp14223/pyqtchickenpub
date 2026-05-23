from PySide6 import QtCore, QtWidgets


class Ui_DetectPage(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(1280, 860)
        MainWindow.setWindowTitle("检测页面")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        rootLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        rootLayout.setContentsMargins(16, 16, 16, 16)
        rootLayout.setSpacing(12)

        topBar = QtWidgets.QHBoxLayout()

        self.btnOpen = QtWidgets.QPushButton("选择文件")
        self.btnStop = QtWidgets.QPushButton("停止")

        self.comboModel = QtWidgets.QComboBox()
        self.comboModel.addItems([
            "rknn/yolov8n.pt",
            "rknn/yolo11n.pt",
            "rknn/yolov8s-seg.pt",
        ])
        self.comboModel.setCurrentIndex(2)
        self.comboModel.setMinimumWidth(220)

        self.radioImage = QtWidgets.QRadioButton("图片模式")
        self.radioVideo = QtWidgets.QRadioButton("视频模式")
        self.radioImage.setChecked(True)

        self.modeGroup = QtWidgets.QButtonGroup(MainWindow)
        self.modeGroup.addButton(self.radioImage)
        self.modeGroup.addButton(self.radioVideo)

        self.labelMode = QtWidgets.QLabel("当前模式: 图片")
        self.labelPath = QtWidgets.QLabel("未选择文件")
        self.labelPath.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        topBar.addWidget(self.btnOpen)
        topBar.addWidget(self.btnStop)
        topBar.addSpacing(12)
        topBar.addWidget(QtWidgets.QLabel("模型:"))
        topBar.addWidget(self.comboModel)
        topBar.addSpacing(12)
        topBar.addWidget(self.radioImage)
        topBar.addWidget(self.radioVideo)
        topBar.addSpacing(16)
        topBar.addWidget(self.labelMode)
        topBar.addStretch(1)
        topBar.addWidget(self.labelPath, 2)
        rootLayout.addLayout(topBar)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)

        self.leftPanel = self._build_image_panel("左侧原图")
        self.rightPanel = self._build_image_panel("右侧检测结果")

        content.addWidget(self.leftPanel, 1)
        content.addWidget(self.rightPanel, 1)
        rootLayout.addLayout(content, 1)

        coordBox = QtWidgets.QGroupBox("banana 分割坐标")
        coordLayout = QtWidgets.QVBoxLayout(coordBox)
        coordLayout.setContentsMargins(12, 18, 12, 12)

        self.textCoords = QtWidgets.QPlainTextEdit()
        self.textCoords.setReadOnly(True)
        self.textCoords.setPlaceholderText("检测到 banana 后，这里会显示分割轮廓坐标。")
        coordLayout.addWidget(self.textCoords)

        rootLayout.addWidget(coordBox, 0)

        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

        self.btnOpen.setMinimumHeight(40)
        self.btnStop.setMinimumHeight(40)

        self._apply_style()

    def _build_image_panel(self, title):
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(12, 18, 12, 12)

        label = QtWidgets.QLabel("等待加载")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(560, 520)
        label.setStyleSheet(
            "QLabel {"
            "border: 2px dashed #6b7280;"
            "background: #111827;"
            "color: #d1d5db;"
            "font-size: 18px;"
            "}"
        )
        layout.addWidget(label)
        return box

    def _apply_style(self):
        self.centralwidget.setStyleSheet(
            """
            QWidget {
                background: #0f172a;
                color: #e5e7eb;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:pressed {
                background: #1e40af;
            }
            QRadioButton {
                spacing: 6px;
            }
            QPlainTextEdit {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #e5e7eb;
                font-family: Consolas, monospace;
            }
            QStatusBar {
                background: #111827;
                color: #9ca3af;
            }
            """
        )

