## "合肥高新杯"心电人机智能大赛 - 提交说明

- 队伍：Just4Fun  
- 成员：qqggg、刘根0924、luguoss
- 时间：2019年10月11日

## 1. 算法简述

- **预处理**：
  - 合并train及testA数据集，移除重复数据；  
  - 根据给出的8个导联，计算III、aVR、aVL及aVF等导联；  
  - 计算数据集的均值及标准差，进而做归一化；
- **模型**：
  - 使用的模型为修改的1维的DenseNet；
- **训练**：
  - 使用scikit-multilearn将合并后的数据集分为5折，  
  - 分别使用5折数据训练模型；
  - 训练过程中使用数据增广，微调ECG信号幅值，微调ECG信号基线位置；
  - 使用F1Loss、FocalLoss及MultiLabelSoftMarginLoss复合损失函数；
  - 使用每一折数据训练完模型后，使用验证集的预测结果进行阈值搜索，进一步提升验证集分类效果；  
- **预测**：
  - 分别使用5个模型及5个搜索出的阈值对testB数据集进行预测；
  - 对预测结果进行投票，若某个类别出现次数>=3，则保留此类别。

## 2. 依赖项
使用```Miniconda```创建环境，安装所需依赖项，依赖项版本如下：

|依赖|版本|说明|
|:----:|:----:|:----:|
|conda|4.5.11|创建环境|
|python|3.7.3||
|tqdm|4.32.1||
|numpy|1.16.4|conda安装|
|scipy|1.3.0||
|pandas|0.24.2||
|pytorch|1.1.0|conda安装(cuda 9.0)|
|scikit-learn|0.21.2||
|scikit-multilearn|0.2.0|pip install scikit-multilearn|

## 3. 运行说明

仅在```Ubuntu 18.04```及```CentOS 7.6```环境下做过测试。
- 切换工作目录至```code```文件夹；
- 运行```./main.sh```进行数据的预处理、模型训练、测试集预测。

## 4. 其他说明
受某些随机变量的影响，可能无法完全复现最佳提交结果。  
在此提供预训练的网络以备所需。  
训练好的网络可到此处下载：  
```
链接：https://pan.baidu.com/s/1Cat7MDYmfP5pNdyO6ZWqZQ  
提取码：ikd5
```
将此文件解压缩至```user_data```文件夹，结构如下：
```
|----- user_data
    |----- models
       |----- 1 (含有模型文件及阈值文件)
       |----- 2 (含有模型文件及阈值文件)
       |----- 3 (含有模型文件及阈值文件)
       |----- 4 (含有模型文件及阈值文件)
       |----- 5 (含有模型文件及阈值文件)
```
在```main.sh```中**仅保留最后做预测的命令**：
```
python ./test/ecg_test.py \
    -s ../data/hf_round1_testB_noDup_rename/testB_noDup_rename \
    -m ../user_data/models \
    -t ../data/hf_round1_subB_noDup_rename.txt \
    -a ../data/hf_round1_arrythmia.txt \
    -o ../prediction_result \
    -g 0
```
运行```main.sh```即可获得最佳提交结果。

---

**封装仓促，若运行过程中出现任何问题，请联系：**
```
quqixun@gmail.com
```