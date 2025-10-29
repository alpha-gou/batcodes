将 iTerm2 的配置文件（如 com.googlecode.iterm2.plist）纳入一个 Git 仓库进行版本控制。然后利用​​硬链接​​的特性，使得原始位置的配置文件与 Git 仓库中的文件保持同步（任意一方的修改都会同步到另一方）

```
# 创建硬链接，将配置文件链接到Git仓库目录
ln ~/Library/Preferences/com.googlecode.iterm2.plist ./
```


