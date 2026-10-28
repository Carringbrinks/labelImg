# labelImg Windows exe 打包说明

## 方式一:直接用已打好的包(推荐)

如果已经拿到 `labelImg-windows.zip`,解压后:

```
labelImg/
├── labelImg.exe          ← 主程序,双击运行
├── _internal/            ← 运行时依赖(不要改动)
└── best.onnx             ← AI 检测模型(可选,放这里则自动识别)
```

- 首次运行若提示模型不存在:把训练好的 `best.onnx` 复制到 `labelImg.exe` 同级目录,或启动后在 **AI 检测 → AI 模型设置** 里手动选择模型路径和类别文件(`_internal/data/ai_classes.txt`)。
- 类别文件 `data/ai_classes.txt`(spine / book shelf / main shelf)已打包进 `_internal/data/`。
- 本包为 **CPU 版 onnxruntime**;如需 GPU 加速(CUDA),见下文"GPU 版"。
- 个人设置保存在 `%USERPROFILE%\.labelImgSettings.pkl`,删除该文件可重置所有设置。

## 方式二:在 Windows 机器上自行打包

前提:安装 [Python 3.10/3.11](https://www.python.org/downloads/),安装时勾选 **Add python.exe to PATH**。

1. 把整个项目文件夹拷贝到 Windows 机器。
2. 把 `best.onnx` 放到项目根目录的 `model/` 子目录(可选,打包时自动带上)。
3. 双击运行 `packaging/build_exe.bat`(或在项目根目录执行)。

输出在 `dist/labelImg/`:

```
dist/labelImg/
├── labelImg.exe
├── _internal/
└── best.onnx        (如果第 2 步放了模型)
```

4. 压缩整个 `dist/labelImg` 文件夹即可分发给其他机器(其他机器无需安装 Python)。

### GPU 版(CUDA)

把 `packaging/requirements-windows.txt` 里的 `onnxruntime>=1.18` 换成
`onnxruntime-gpu>=1.18` 后重新打包即可;目标机器需安装对应的
[CUDA](https://developer.nvidia.com/cuda-downloads) 和 cuDNN。

### 单文件 exe(onefile)

当前是 **onedir** 模式(启动快、更稳定)。如需单文件 exe,在 spec 的
`EXE(...)` 中加 `a.binaries + a.datas`,并把 `exclude_binaries` 改为 `True`,
去掉 `COLLECT` 段。注意:单文件每次启动都要解压到临时目录(约 300MB),
启动慢数秒,且被杀毒软件误报的概率更高。

## 常见问题

- **双击没反应**:`dist/labelImg/labelImg.exe` 是窗口程序,没有控制台输出。
  可在 cmd 里运行 `labelImg.exe` 查看报错信息。
- **AI 检测报 "DLL 初始化例程失败" (动态链接库(DLL)初始化例程失败)**:
  onnxruntime 需要 VC++ 2022 运行库(14.40+),而 Python 安装器自带的
  运行库偏旧,被打进包后会导致此错误。用本目录的 `build_exe.bat`
  重新打包即可(脚本会自动安装官方 VC++ 2022 运行库并更新到输出目录)。
  对已打好的包,也可以在目标机器安装一次
  [VC++ 2015-2022 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe) 临时解决。
- **检测按钮提示模型不存在**:按方式一的说明放置 best.onnx 或在设置里选择。
- **SmartScreen 提示**:PyInstaller 打包的程序未签名,点"仍要运行"即可。
