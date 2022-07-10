<!--
 * @Author: mrk-lyz mrk_lanyouzi@yeah.net
 * @Date: 2022-03-06 15:05:20
 * @LastEditTime: 2022-07-10 20:46:47
 * @FilePath: /pd/Readme.md
 * @Description: 
 * 
 * Copyright (c) 2022 by mrk-lyz mrk_lanyouzi@yeah.net, All Rights Reserved. 
-->
# Readme

## File Directory

- config：配置参数
- core：数据集和训练代码
- medication_detection：药物依从性
- subject_detection：主体识别

## Subject Detection

- 使用记录数量最多的50个subject的数据，共计6805个record（每个人的记录在46~90不等），104436个cycle。
- 训练集89492条(90%)，测试集包含9944条（10%），测试集为100（条）×50(人)，样本较为平衡。
- 在实验中设置了WeightSampler和FocalLoss作为对比，具体如下表所示。

|**Weighted Sampler**|   **Loss**  | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Remark**     |
|:------------------:|:-----------:|:------------:|:-------------:|:----------:|:------------:|:--------------:|
|       True         | Focal Loss  |      0.70    |    0.73       |     0.70   |      0.70    | bs=128,lr=1e-4 |
|       True         | Focal Loss  |      0.72    |    0.75       |     0.72   |      0.72    | bs=128,lr=1e-2 |
|       True         |  CE Loss    |      0.67    |    0.69       |     0.67   |      0.67    | bs=128,lr=1e-4 |
|       True         |  CE Loss    |      0.72    |    0.75       |     0.72   |      0.72    | bs=128,lr=1e-3 |

## Medication Intake Detection

![framework](https://raw.githubusercontent.com/lanyouzi/images/master/img/202207102041746.png)

- 使用数据量最多的前50个用户的数据，随机挑选10%（5人）作为训练。
- 训练样本对是按照每个患者的数据随机采样构造。
- 使用SGD和AdamW作为优化器，AUC维持在0.54左右。
