# 上传 GitHub

推荐仓库名称：aces-echo-segmentation。

## 推荐：GitHub Desktop

本包包含较多逐运行记录，使用 GitHub Desktop 上传整个目录比较方便。

1. 解压 ACES_GitHub_Ready.zip，进入 ACES_GitHub 文件夹，确认能看到 README.md、aces、data、results、tests。
2. 打开 GitHub Desktop，登录自己的 GitHub。
3. 选择 File → New repository，建立空仓库，选择要保存的位置。此时可以先不选择开源许可。
4. 将解压目录里的全部内容复制到新仓库根目录；包括 .gitignore、.gitattributes 和 .github 文件夹。
5. 确认 README.md 位于仓库根目录，而不是再套一层 ACES_GitHub。
6. 在 GitHub Desktop 填写提交说明，例如 Initial code and experiment records，点击 Commit。
7. 点击 Publish repository，选择公开或私有后发布。

只复制本整理包即可。原电脑的 .venv、runtime、下载影像、prepared、runs、best.pt、last.pt、日志不用复制到公开仓库。

## 已会用 Git 的方式

在解压后、能看到 README.md 的目录打开终端：

    git init
    git add .
    git commit -m "Add ACES code and experiment records"
    git branch -M main

在 GitHub 创建空仓库后，按页面给出的实际远程地址执行：

    git remote add origin 你的实际仓库地址
    git push -u origin main

示例中的中文地址必须换成 GitHub 创建仓库后给你的实际地址。

## 网页上传

可以使用 Add file → Upload files。GitHub 网页每批最多 100 个文件，本包超过这个数量，需要分批；单个文件均小于网页单文件限制。用 Desktop 上传可以省去逐批处理。

上传展开后的项目文件，方便读者查看源码和README。压缩包也可作为 Release 附件保存。

## 上传后

- 打开仓库首页，确认 README 正常显示。
- 打开 Actions，查看 records 和 cpu-tests 的结果。
- 论文中的代码链接填写你最终创建的实际仓库地址。
- 若准备允许他人修改/再分发代码，再由著作权人选择许可并添加 LICENSE；本次整理没有代替作者授权。

官方说明：https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
