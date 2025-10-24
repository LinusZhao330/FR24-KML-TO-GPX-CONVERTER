# FR24 KML to GPX 工具

## 简介
FR24_to_GPX 用于将 FlightRadar24 导出的 KML 航迹文件转换为通用 GPX 轨迹，并提供轨迹预览、插值补点、时间修正以及导航规划等功能。项目同时包含命令行工作流和基于 Tkinter 的图形界面，适合批处理、交互式预览以及封装成可执行程序。

## 功能亮点
- **KML -> GPX 转换**：解析 FR24 导出的 Route 图层，保留时间、经纬度与高度信息。
- **时间插值与补点**：按照自定义时间间隔对轨迹进行线性插值，平滑缺失点。
- **目标日期修正**：以 UTC+8 为基准重新设定轨迹日期，方便复用飞行记录。
- **图形界面预览**：内置日志、统计摘要、地图预览（TkinterMapView 或 folium 浏览器版），并可在导出后打开文件所在位置。
- **航路规划（实验特性）**：基于 OpenStreetMap Nominatim 与 OSRM，按起点、终点和途经点生成地面路线并导出 GPX。
- **交互式 CLI**：使用 click，即使没有给出参数也会通过提示引导输入，便于集成脚本或批处理。

## 环境要求
- Python 3.10 及以上（代码使用了 3.10 的联合类型语法）。
- 核心依赖：`click`, `requests`。
- 可选依赖：
  - `tkinter`（Python 自带，部分发行版需单独安装）。
  - `tkintermapview`（提供嵌入式地图）。
  - `folium`（当没有 TkinterMapView 时使用浏览器地图预览）。
- 网络访问：航路规划、地理编码以及在线地图都需要能够访问外网。

## 安装
1. 克隆项目：
   ```powershell
   git clone https://github.com/zzk90/FR24_to_GPX.git
   cd FR24_to_GPX
   ```
2. 建议创建虚拟环境：
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. 安装依赖：
   ```powershell
   pip install -U click requests
   # 可选：安装地图相关依赖
   pip install -U tkintermapview folium
   ```

## 快速开始
### 命令行
```powershell
python converter.py --kml path\to\flight.kml --gpx flight.gpx --interval 60 --date 2024-08-15
```

常用选项：

| 选项 | 说明 |
| --- | --- |
| `--kml PATH` | 输入 KML 文件路径（必填，若缺省会提示交互输入）。 |
| `--gpx PATH` | 输出 GPX 文件路径，缺省为与源文件同名。 |
| `--interval SEC` | 插值间隔（秒），`0` 表示不补点，默认 60。 |
| `--date YYYY-MM-DD` | 以 UTC+8 为基准重写全部时间戳，不填写则保留原始时间。 |
| `--gui` / `--cli` | 强制启动 GUI 或 CLI；打包的可执行文件默认进入 GUI。 |

转换完成后命令行会输出生成文件路径，并在需要时显示警告信息（例如无法解析高度或日期）。

### 图形界面
```powershell
python converter.py --gui
```
或在打包后的可执行文件上直接双击运行。

界面主要功能：
- 使用 “Select KML” 选择文件，点击 “Preview” 先分析航迹、填充目标日期并更新摘要。
- “Interpolation interval” 控制补点间隔，“Target date” 可手动调整。
- “Convert to GPX” 执行转换，完成后可通过 “Open output location” 打开输出目录。
- “Map preview” 标签页展示轨迹；当没有 `tkintermapview` 时会改用浏览器打开 folium 地图。
- 底部日志窗口记录状态与警告信息，方便排查问题。

## 航路规划（实验特性）
图形界面的 “Navigation planner” 区域可以：
1. 输入起点、终点和可选途经点（每行一个）。
2. 设置导出采样间隔。
3. 点击 “Plan route” 调用 Nominatim 与 OSRM 获取道路路线。
4. 在地图中预览结果，并使用 “Export GPX” 导出。

> 该功能需要稳定的网络连接，频繁请求可能被服务限流。

## 常见问题
- **无法打开 GUI / 缺少 Tkinter**：在 Linux 上安装 `python3-tk` 或发行版提供的 Tk 包。
- **界面没有地图**：安装 `tkintermapview`；若仍无法显示，可安装 `folium` 使用浏览器预览。
- **航路规划失败**：检查网络状况，稍后重试，或减少请求频率；必要时直接在 CLI 中转换 KML。
- **高度数据缺失**：当 FR24 KML 中缺少高度时，程序会尝试从 description 提取；若仍失败，可在生成的 GPX 中手工调整。

## 开发与构建
- 主入口为 `converter.py`，包含 CLI 和 GUI 的全部逻辑。
- 仓库提供了 `converter.spec` 与 `FR24Converter.spec`，可用 PyInstaller 构建可执行文件：
  ```powershell
  pyinstaller converter.spec
  ```
- `update_dataclasses.py` 用于在调整数据结构时批量更新类型定义。
- 欢迎提交 Issue 或 PR，例如改进插值策略、增加地图后端或完善文档。

---

如果在使用中遇到问题或有新的想法，欢迎记录日志并反馈，共同完善这个工具。
