# 如何使用 VS Code 连接 AWS EC2 实例

## 📋 前置准备

在开始之前，确保你有：
- ✅ 一个正在运行的 AWS EC2 实例
- ✅ EC2 实例的公网 IP 地址（例如：`100.48.50.178`）
- ✅ 用于连接的 `.pem` 密钥文件（例如：`makbook air4.pem`）
- ✅ 安装了 VS Code

---

## 步骤 1️⃣：安装 VS Code 扩展

### 1.1 安装 Remote - SSH 扩展

1. 打开 VS Code
2. 点击左侧的**扩展**图标（或按 `Cmd+Shift+X`）
3. 搜索 `Remote - SSH`
4. 点击 **Install** 安装

![Remote SSH Extension](https://code.visualstudio.com/assets/docs/remote/ssh/ssh-extension.png)

---


### 2.1 打开终端（Terminal）

在 VS Code 中按 `Ctrl+`` 或者从菜单选择 **Terminal > New Terminal**

### 2.2 设置密钥文件权限

密钥文件就是发给你的macbook air4.pem

```bash
# 进入密钥文件所在目录
cd ~/Downloads #有可能不在这 看你下哪了

# 设置正确的权限（必须是 400 或 600）
chmod 400 "makbook air4.pem"
```

**⚠️ 重要**：如果不设置权限，SSH 连接会被拒绝！

---

## 步骤 3️⃣：配置 SSH 连接

### 3.1 打开 SSH 配置文件

1. 在 VS Code 中按 `Cmd+Shift+P` 打开命令面板
2. 输入 `Remote-SSH: Open SSH Configuration File`
3. 选择你的配置文件（通常是 `~/.ssh/config`）

### 3.2 添加 EC2 连接配置

在配置文件中添加以下内容：
（最后一行的路径记得换成密钥文件的路径）

```ssh_config
Host 5190-instance
  HostName 100.48.50.178
  User ubuntu
  IdentityFile /Users/feiyanj/Downloads/makbook air4.pem
```


**参数说明：**
- `Host`：连接的别名（可以自定义）
- `HostName`：EC2 实例的公网 IP
- `User`：登录用户名（Ubuntu 系统通常是 `ubuntu`，Amazon Linux 是 `ec2-user`）
- `IdentityFile`：`.pem` 密钥文件的完整路径

**💡 提示**：如果路径中有空格，记得用反斜杠转义：
```
IdentityFile /Users/feiyanj/Downloads/makbook\ air4.pem
```

### 3.3 保存配置文件

按 `Cmd+S` 保存

---

## 步骤 4️⃣：连接到 EC2 实例

### 4.1 启动连接

1. 按 `Cmd+Shift+P` 打开命令面板
2. 输入 `Remote-SSH: Connect to Host`
3. 选择你刚才配置的 `5190-instance`

### 4.2 等待连接建立

- 首次连接会询问 "Are you sure you want to continue?"，选择 **Continue**
- VS Code 会在远程服务器上安装必要的组件
- 连接成功后，左下角会显示 `SSH: 5190-instance`

![SSH Connected](https://code.visualstudio.com/assets/docs/remote/ssh/ssh-statusbar.png)

---

## 步骤 5️⃣：在远程服务器上工作

### 5.1 打开远程文件夹

1. 点击 **File > Open Folder**（或按 `Cmd+O`）
2. 输入项目路径，例如：`/home/ubuntu/cis5190_project_b`
3. 点击 **OK**

### 5.2 使用远程终端

- 按 `Ctrl+`` 打开终端
- 这个终端现在运行在 EC2 实例上！
- 可以直接运行命令，例如：


### 5.3 编辑远程文件

- 在左侧文件浏览器中打开任何文件
- 编辑后按 `Cmd+S` 保存
- 更改会直接保存到远程服务器上

---



## 📝 快速参考

### SSH 配置模板

```ssh_config
Host <别名>
  HostName <EC2公网IP>
  User <用户名>
  IdentityFile <密钥文件路径>
```

### VS Code 快捷键

| 操作 | 快捷键 |
|------|--------|
| 打开命令面板 | `Cmd+Shift+P` |
| 打开终端 | ``Ctrl+` `` |
| 打开文件夹 | `Cmd+O` |
| 保存文件 | `Cmd+S` |


