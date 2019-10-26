Docker提交记录

```dockerfile
# Build image
sudo docker build -t registry.cn-shanghai.aliyuncs.com/hf_ecg/hf_ecg_submit:[镜像版本] .
# Login
sudo docker login --username=quqixun@gmail.com registry.cn-shanghai.aliyuncs.com
# Push image
sudo docker push registry.cn-shanghai.aliyuncs.com/hf_ecg/hf_ecg_submit:[镜像版本]
```

|文件夹|镜像版本|镜像地址|模型|预训练|其他|
|---------|-----------|------------|-----|---------|------|
|docker1|0.0.6| registry.cn-shanghai.aliyuncs.com/hf_ecg/hf_ecg_submit:0.0.3|small|无|无|
|docker2|0.0.7| registry.cn-shanghai.aliyuncs.com/hf_ecg/hf_ecg_submit:0.0.4|small|有|无|
