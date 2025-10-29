# 一些配置的保存和同步

## iTerm2
将 iTerm2 的配置文件（如 com.googlecode.iterm2.plist）纳入一个 Git 仓库进行版本控制。

然后利用硬链接的特性，使得原始位置的配置文件与 Git 仓库中的文件保持同步（任意一方的修改都会同步到另一方）

```
# 创建硬链接，将配置文件链接到Git仓库目录
ln ~/Library/Preferences/com.googlecode.iterm2.plist ./
```

## VS Code
使用官方设置同步 (最推荐)

这是最无缝、高效的方案，能同步包括已安装扩展列表在内的几乎所有个性化设置。

开启同步：在 VS Code 中，按下 Ctrl+Shift+P(Windows/Linux) 或 Cmd+Shift+P(Mac) 打开命令面板。输入并选择 "Settings Sync: Turn On"。接着按提示使用你的 GitHub 账户登录，并选择需要同步的项目（如设置、键盘快捷键、扩展等）。

在新设备上恢复：在新设备的 VS Code 上同样开启设置同步并登录同一个 GitHub 账户，配置会自动下载并应用


## Sublime Text
Sublime Text 主要依赖手动备份配置文件。

定位配置文件：其用户配置保存在 Packages/User/目录下，具体路径因操作系统而异 ：

```
Windows: `C:\Users[用户名]\AppData\Roaming\Sublime Text 3\Packages\User`
macOS: ~/Library/Application Support/Sublime Text 3/Packages/User/
Linux: ~/.config/sublime-text-3/Packages/User/
（注意：对于 Sublime Text 4，路径中的版本号可能变为 "Sublime Text" 或 "sublime-text"）
```

直接使用如下命令即可
```
cp ./Default\ \(OSX\).sublime-keymap ~/Library/Application\ Support/Sublime\ Text/Packages/User/  
```
