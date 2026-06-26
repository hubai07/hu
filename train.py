from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=50,
        batch=8,
        # 数据增强（适度增强，避免过度）
        mosaic=0.3,  # 适度mosaic增强（0.5比0更适合小数据集）
        hsv_h=0.015,  # 色调增强（小幅）
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=8.0,  # 旋转角度（小幅）
        translate=0.05,  # 平移（小幅）
        scale=0.05,  # 缩放（小幅）
        fliplr=0.5,  # 水平翻转
        flipud=0.0,  # 垂直翻转（根据数据集决定）
        mixup=0.0,  # 关闭mixup（小数据集易过拟合）
        copy_paste=0.1,  # 关闭copy_paste
        # 优化器配置
        optimizer="AdamW",  # AdamW适合小数据集
        lr0=0.0005,  # 初始学习率（小数据集调低）
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,  # 动量
        weight_decay=0.001,  # 权重衰减（防止过拟合）
        warmup_epochs=3,  # 预热轮数（小数据集3轮足够）
        warmup_momentum=0.8,  # 预热动量
        warmup_bias_lr=0.01,  # 预热偏置学习率
        # 正则化（关键：防止过拟合）
        dropout=0.2,  # dropout层（小数据集必备）
        label_smoothing=0.1,  # 标签平滑
        freeze=10,
        workers=4,
        project="runs/train",
        name="exp",
    )
