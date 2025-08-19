## 关键文件说明

### main_binary.py
这是训练和评估量化神经网络的主脚本。

**主要功能：**
- 支持训练MobileNet网络架构（cifar10，cifar100，imagenet数据集等）
- 实现权重和激活的亚字节量化
- 支持不同的量化策略：
  - 每层量化（MixPL）
  - 每通道量化（MixPC）
  - 混合量化
- 内存驱动的自动位宽选择
- 批归一化折叠到卷积层
- 导出量化后纯整数部署网络

**训练命令示例：**
python3 main_binary.py -a mobilenet --mobilenet_width 0.25 --mobilenet_input 160 --save Imagenet/mobilenet_cifar100_4bit_icn160PLPACTwidth0.5 --data cifar100 --type_quant "PerLayerAsymPACT"   --weight_bits 4 --activ_bits 4 --activ_type learned --gpus 0 -j 8 --epochs 300 -b 64 --save_check --quantizer --batch_fold_delay 50 --batch_fold_type ICN

### weight.py
此脚本处理将量化模型参数导出到微控制器部署用的C头文件。

**主要功能：**
- 从检查点文件读取量化参数
- 将4位权重转换为8位格式（通过组合两个4位值）
- 生成带量化参数的每层头文件
- 将所有层头文件合并为单个`merged_layers.h`文件
- 处理量化偏移量（Z_w、Z_in、Z_out）
- 将浮点参数转换为定点格式以实现高效硬件实现

### gengrate.py
此脚本生成用于在微控制器上部署量化模型的C代码。

**主要功能：**
- 计算最大缓冲区需求以实现内存分配
- 生成用于调用量化层的算子映射
- 支持标准卷积和深度可分离卷积
- 处理填充、步长和其他卷积参数
- 与`merged_layers.h`中的量化参数集成
