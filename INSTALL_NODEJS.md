# Node.js 安装指南

## 方法 1：使用 winget（推荐）

打开 PowerShell（**以管理员身份运行**），然后执行：

```powershell
winget install --id OpenJS.NodeJS.20 --silent --accept-package-agreements --accept-source-agreements
```

## 方法 2：手动下载

1. 访问：https://nodejs.org/
2. 下载 **LTS 版本**（推荐 v20.x）
3. 运行安装程序
4. 勾选 "Automatically install necessary dependencies"（可选）
5. 完成安装

## 验证安装

安装完成后，打开新的 PowerShell 窗口，执行：

```powershell
node --version
npm --version
```

应该显示类似：
```
v20.20.2
10.8.1
```

## 安装后运行项目

```powershell
cd mini-world-model/threejs-game-skills/threejs-spatial-walker
npm install
npm run dev
```

然后访问 http://localhost:5173

---

## 故障排除

### 问题 1：winget 需要管理员权限

右键点击 PowerShell → "以管理员身份运行"

### 问题 2：安装后 node 命令找不到

1. 关闭并重新打开 PowerShell
2. 或者重启电脑

### 问题 3：npm install 失败

```powershell
npm cache clean --force
npm install
```

### 问题 4：端口被占用

```powershell
npm run dev -- --port 5174
```
