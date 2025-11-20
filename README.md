# WebGL 流体模拟 | Fluid Dynamics Simulation

一个炫酷的 GPU 加速流体动力学模拟系统，使用 WebGL 2 和 GLSL 着色器实现实时 Navier-Stokes 流体物理。

## ✨ 特性

- 🌊 **实时流体物理** - GPU 加速的 Navier-Stokes 方程求解
- 🎨 **多种配色方案** - 霓虹、彩虹、火焰、海洋四种主题
- 🎮 **交互式控制** - 鼠标/触摸拖动创造流体效果
- 💎 **赛博朋克 UI** - Glassmorphism 玻璃态设计
- ⚡ **高性能** - 60 FPS 流畅运行

## 🚀 在线演示

访问：[https://uttgeorge.github.io/antigravity_test_1/](https://uttgeorge.github.io/antigravity_test_1/)

## 🎮 使用方法

1. **创造流体** - 在画布上拖动鼠标
2. **切换配色** - 点击右侧面板的配色按钮
3. **调节参数** - 使用滑块实时调整流体行为
   - 粘度 (Viscosity) - 控制流体厚度
   - 扩散 (Diffusion) - 控制颜色扩散
   - 压力 (Pressure) - 计算质量
   - 涡度 (Curl) - 漩涡强度
   - 笔刷大小 (Brush Size) - 影响范围
4. **查看性能** - 点击"统计 Stats"按钮
5. **重置画布** - 点击"清除 Clear"按钮

## 🛠️ 技术栈

- **WebGL 2** - 图形渲染和 GPU 计算
- **GLSL Shaders** - 流体物理计算
- **Vanilla JavaScript** - 无框架依赖
- **Modern CSS** - Glassmorphism 设计

## 📦 本地运行

```bash
# 克隆仓库
git clone https://github.com/uttgeorge/antigravity_test_1.git
cd antigravity_test_1

# 启动本地服务器
python3 -m http.server 8000

# 访问 http://localhost:8000
```

## 🎨 配色方案

- **霓虹 (Neon)** - 青色、品红、紫色
- **彩虹 (Rainbow)** - 全光谱色彩
- **火焰 (Fire)** - 红、橙、黄渐变
- **海洋 (Ocean)** - 深蓝、青色、绿松石

## 📝 文件结构

```
.
├── index.html    # HTML 结构
├── style.css     # 样式表
├── fluid.js      # 流体模拟引擎
└── README.md     # 项目说明
```

## 🌐 浏览器支持

- ✅ Chrome/Edge
- ✅ Firefox
- ✅ Safari
- ⚠️ 需要支持 WebGL 2

## 📄 许可证

MIT License

## 🙏 致谢

基于 GPU 加速的 Navier-Stokes 流体模拟算法。

---

**享受流体艺术！** 🌊✨
