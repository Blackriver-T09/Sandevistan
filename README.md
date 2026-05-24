# Sandevistan - 赛博朋克视频特效

实现《赛博朋克：边缘行者》中的"斯安威斯坦"时间停止特效。

## 当前功能

✅ **实时轮廓高亮** - 自动识别视频中的人物并高亮轮廓  
✅ **半透明填充** - 为识别的目标添加彩色半透明填充  
🚧 **运动残影** - 即将实现

## 项目结构

```
Sandevistan/
├── config/
│   └── effect_config.yaml    # 效果参数配置
├── src/
│   ├── config.py             # 配置加载器
│   ├── detector.py           # YOLOv8 人体检测/分割
│   ├── renderer.py           # 轮廓渲染器
│   └── video_processor.py    # 视频处理器
├── main.py                   # 主程序入口
└── requirements.txt          # Python 依赖
```

## 安装

```bash
# 激活 conda 环境
conda activate sandevistan

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```bash
# 基础用法
python main.py test.mp4

# 使用自定义配置
python main.py test.mp4 config/custom_config.yaml
```

## 配置参数

编辑 `config/effect_config.yaml` 来调整效果：

```yaml
contour:
  color: [0, 255, 255]      # 轮廓颜色 (BGR)
  thickness: 3              # 线宽

fill:
  color: [255, 200, 0]      # 填充颜色 (BGR)
  alpha: 0.4                # 透明度

model:
  name: "yolov8n-seg.pt"    # 模型大小
  conf: 0.5                 # 置信度阈值
  device: "cpu"             # cuda/cpu
```

## 技术栈

- **YOLOv8-seg** - 实例分割
- **OpenCV** - 图像处理
- **PyTorch** - 深度学习推理

## 输出

处理后的视频将保存为 `<原文件名>_highlighted.<扩展名>`
