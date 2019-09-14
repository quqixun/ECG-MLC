# HFECG

文件夹结构
- data
	- train_txt: 文件夹，训练集txt (这里没有，自己加)
	- train_npy: 文件夹，txt转成npy (这里没有，自己加)
	- textA_txt: 文件夹，测试集A的txt (这里没有，自己加)
	- textA_npy: 文件夹，测试集A的txt转成npy (这里没有，自己加)
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