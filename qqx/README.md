# HFECG

["合肥高新杯"心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction)

## 文件夹结构

- data
	- train_txt: 文件夹，训练集txt
	- train_npy: 文件夹，txt转成npy (这里没有，自己加)
	- textA_txt: 文件夹，测试集A的txt (这里没有，自己加)
	- textA_npy: 文件夹，测试集A的txt转成npy
	- hf_round1_arrythmia.txt: 标签列表
	- hf_round1_label.txt: 训练集标签
	- hf_round1_subA.txt: 测试集A列表
	- train.csv: 训练集标签转成one hot的csv
- src
	- baseline
		- train.py, loader.py, adabound.py, params.json: 训练脚本和参数
		- run_resnet.sh, run_tcn.sh, run_mstcn.sh: 运行训练脚本
		- resnet.py, tcn.py: 模型
		- test.py, predict.sh: 做推断
		- utils.py: loss function
	- utils
		- gen_labels.py: 训练集标签txt转成one hot的csv
		- prep_ecg.py: 预处理，降采样到100Hz，并计算mean和std
		- plot_ecg.py: 画图
- outputs: 预测结果，提交文件

## 提交结果
|NO.|Model|Submission|TestA|TestB|
|:-:|:---:|:--------:|:---:|:---:|
|1|[ResNet](https://github.com/quqixun/HFECG/blob/master/qqx/src/baseline/resnet.py#L65)|[File](./outputs/baseline-resnet/baseline-resnet.txt)|0.734|-|
|2|[ResNet](https://github.com/quqixun/HFECG/blob/master/qqx/src/baseline/resnet.py#L65) Ensemble|[File](./outputs/baseline-resnet/baseline-resnet-ensemble.txt)|0.752|-|
|3|12 Leads Input + F1 Loss|[File](./outputs/i2-12-resnet-mbw/i2-12-resnet-ensemble.txt)|0.8119|-|
|4|Search Threshold|[File](./outputs/st-12-resnet/st-12-resnet-ensemble.txt)|0.8131|-|
|5|HRV Features + 100Hz Input|[File](./outputs/hrv-100-12-resnet/hrv-100-12-resnet-ensemble.txt)|0.8136|-|
|6|HRV Features + 200Hz Input|[File](./outputs/hrv-200-12-resnet/hrv-200-12-resnet-ensemble.txt)|0.8193|-|
