from PySide6 import QtCore, QtWidgets


class Ui_DetectPage(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(1320, 900)
        MainWindow.setMinimumSize(980, 680)
        MainWindow.setWindowTitle("检测页面")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        root_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(10)

        self.btnOpen = QtWidgets.QPushButton("选择文件")
        self.btnCapture = QtWidgets.QPushButton("拍照检测")
        self.btnStop = QtWidgets.QPushButton("停止")

        self.comboModel = QtWidgets.QComboBox()
        self.comboModel.addItems(
            [
                "rknn/yolov11s-seg.pt",
                "rknn/yolov8sseg.pt",
                "rknn/yolov8n.pt",
                "rknn/yolo11n.pt",
            ]
        )
        self.comboModel.setCurrentIndex(0)
        self.comboModel.setMinimumWidth(220)

        self.radioImage = QtWidgets.QRadioButton("图片模式")
        self.radioCamera = QtWidgets.QRadioButton("摄像头模式")
        self.radioImage.setChecked(True)

        self.modeGroup = QtWidgets.QButtonGroup(MainWindow)
        self.modeGroup.addButton(self.radioImage)
        self.modeGroup.addButton(self.radioCamera)

        self.comboCamera = QtWidgets.QComboBox()
        for index in range(4):
            self.comboCamera.addItem(f"USB 摄像头 {index}", index)
        self.comboCamera.setMinimumWidth(140)

        self.comboFlip = QtWidgets.QComboBox()
        self.comboFlip.addItem("不翻转", "none")
        self.comboFlip.addItem("水平翻转", "horizontal")
        self.comboFlip.addItem("垂直翻转", "vertical")
        self.comboFlip.addItem("水平+垂直", "both")
        self.comboFlip.setMinimumWidth(120)

        self.labelMode = QtWidgets.QLabel("当前模式: 图片")
        self.labelPath = QtWidgets.QLabel("未选择文件")
        self.labelPath.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.labelPath.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

        top_bar.addWidget(self.btnOpen)
        top_bar.addWidget(self.btnCapture)
        top_bar.addWidget(self.btnStop)
        top_bar.addSpacing(10)
        top_bar.addWidget(QtWidgets.QLabel("模型:"))
        top_bar.addWidget(self.comboModel)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.radioImage)
        top_bar.addWidget(self.radioCamera)
        top_bar.addSpacing(10)
        top_bar.addWidget(QtWidgets.QLabel("摄像头:"))
        top_bar.addWidget(self.comboCamera)
        top_bar.addWidget(QtWidgets.QLabel("翻转:"))
        top_bar.addWidget(self.comboFlip)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.labelMode)
        top_bar.addStretch(1)
        top_bar.addWidget(self.labelPath, 2)
        root_layout.addLayout(top_bar)

        self.leftPanel = self._build_image_panel("左侧原图 / 摄像头预览")
        self.rightPanel = self._build_image_panel("右侧检测结果")

        self.contentSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.contentSplitter.setChildrenCollapsible(False)
        self.contentSplitter.addWidget(self.leftPanel)
        self.contentSplitter.addWidget(self.rightPanel)
        self.contentSplitter.setStretchFactor(0, 1)
        self.contentSplitter.setStretchFactor(1, 1)

        coord_box = QtWidgets.QGroupBox("chicken 分割坐标")
        coord_layout = QtWidgets.QVBoxLayout(coord_box)
        coord_layout.setContentsMargins(12, 18, 12, 12)

        self.textCoords = QtWidgets.QPlainTextEdit()
        self.textCoords.setReadOnly(True)
        self.textCoords.setPlaceholderText(
            "拍照或图片检测完成后，这里会显示 chicken 的分割轮廓坐标和端点信息。"
        )
        coord_layout.addWidget(self.textCoords)

        self.mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.addWidget(self.contentSplitter)
        self.mainSplitter.addWidget(coord_box)
        self.mainSplitter.setStretchFactor(0, 5)
        self.mainSplitter.setStretchFactor(1, 2)
        root_layout.addWidget(self.mainSplitter, 1)

        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

        self.btnOpen.setMinimumHeight(40)
        self.btnCapture.setMinimumHeight(40)
        self.btnStop.setMinimumHeight(40)

        self._apply_style()

    def _build_image_panel(self, title):
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(12, 18, 12, 12)

        label = QtWidgets.QLabel("等待加载")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
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
            QPushButton:disabled {
                background: #475569;
                color: #cbd5e1;
            }
            QComboBox, QPlainTextEdit {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #e5e7eb;
                padding: 6px 8px;
            }
            QRadioButton {
                spacing: 6px;
            }
            QPlainTextEdit {
                font-family: Consolas, monospace;
            }
            QStatusBar {
                background: #111827;
                color: #9ca3af;
            }
            QSplitter::handle {
                background: #1e293b;
            }
            QSplitter::handle:horizontal {
                width: 8px;
            }
            QSplitter::handle:vertical {
                height: 8px;
            }
            """
        )
