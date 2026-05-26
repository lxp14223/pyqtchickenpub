# github连接不上

清除 Git 的代理设置，让其直接连接网络进行操作。

PS D:\code\git\pyqtchicken> git config --global --unset http.proxy
PS D:\code\git\pyqtchicken> git config --global --unset https.proxy

80%都是代理没有设置好，开启了梯子，但是git工具没有设置相关的代理
导致git其实并没有走代理，一般输入以下命令设置代理即可

git config --global http.proxy 127.0.0.1:7890
git config --global https.proxy 127.0.0.1:7890

# 文件列表

###########目录结构描述
├── Readme.md  
├── icon  
├── rknn  
├── video  
├── detect_page_app.py //程序入口  
├── detect_page_ui.py //ui界面  
├── pca_axis_utils.py //pca方法  
├── skeleton_endpoint_utils.py //骨架方法  
└── tools

###########V1.0.0 版本内容更新

1. 新功能
2. 添加pca 骨架算法
