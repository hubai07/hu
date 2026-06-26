import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, \
    QFileDialog
from PyQt5.QtGui import QPixmap,QFont,QImage
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #窗口初始化
        self.setWindowTitle("目标检测")
        #窗口位置，还有大小
        self.setGeometry(300, 150, 1000, 700)
        self.timer = QTimer()#定时器
        #加载模型
        self.model = YOLO('./runs/train/exp/weights/best.pt')
        #创建主布局
        self.label = QLabel(self)#创建显示图片的QLabel控件
        self.label.setAlignment(Qt.AlignCenter)#设计文本居中
        self.label.setText("请选择图片或者视频进行检测")
        self.label.setFont(QFont("Arial", 16))#设置字体的大小
        #设置背景颜色及边框
        self.label.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray;")
        self.label.setFixedHeight(500)#显示区域的高度
        #3个按钮
        self.img_button = QPushButton("选择图片")
        self.video_button = QPushButton("选择视频")
        self.stop_button = QPushButton("停止视频")
        self.camera_button =QPushButton("摄像头")
        self.stop_button.setEnabled(False)
        #把按钮放置到相应的位置
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.img_button)
        button_layout.addWidget(self.video_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.camera_button)
        #主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(button_layout)
        #包装布局到QWidget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)#设置改容器为主窗口的中心部件

        #图片的上传按钮的点击事件
        self.img_button.clicked.connect(self.handle_image)#图片上传按钮
        self.video_button.clicked.connect(self.handle_video)#视频上传按钮
        self.camera_button.clicked.connect(self.handle_camera)#摄像头按钮
        self.stop_button.clicked.connect(self.handle_stop)#视频停止按钮
        self.timer.timeout.connect(self.handle_video_frame)#定时器用于视频播放

    #进行图片处理
    def handle_image(self):
        #打开文件对话框选择图片
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","","Image Files (*.jpg *.jpeg *.png )")
        if path:
            #读取图片  BGR格式
            img = cv2.imread(path)
            #使用模式进行检测
            result_img = self.model(img)
            #获取模型的结果
            result = result_img[0].plot() #获取绘制后的结果
            #把BBGR格式转RGB
            rgb = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            #获取图片的宽度、高度、通道数
            h, w, ch = rgb.shape
            #将处理后的图片(numpy)转成QImage格式
            qimg = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
            #将QImage转换为QPixmap
            pixmap = QPixmap.fromImage(qimg).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            #在QLabel中显示出来
            self.label.setPixmap(pixmap)
    def handle_video(self):
        #处理加载视频并播放,path是视频路径
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov )")
        if path:
            self.cap = cv2.VideoCapture(path)#用opencv读取视频
            self.timer.start(30)#启动定时器，每隔30毫秒触发一次
            self.stop_button.setEnabled(True)#启动停止按钮
    def handle_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.stop_button.setEnabled(True)

    def handle_video_frame(self):
        ret, frame = self.cap.read()#读取视频的帧
        if not ret:
            #如果没有读到帧，停止定时器
            self.timer.stop()
            self.cap.release()#释放视频捕获对象
            self.stop_button.setEnabled(False)
            return
        #使用模型进行预测
        result_img = self.model(frame)
        result = result_img[0].plot()#获取预测结果
        rgb = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)#BGR转RGB
        h, w, ch = rgb.shape#获取图片的宽度、高度、通道数
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)#将处理后的图片(numpy)转成QImage格式
        # 将QImage转换为QPixmap
        pixmap = QPixmap.fromImage(qimg).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)# 在QLabel中显示出来

    def handle_stop(self):
        self.timer.stop()#停止定时器
        self.cap.release()#释放视频捕获对象
        self.stop_button.setEnabled(False)

#qt程序的入口
app = QApplication(sys.argv)
#创建主窗口的实例
window = MainWindow()
#显示出来
window.show()
#启动事件循环
sys.exit(app.exec_())
























