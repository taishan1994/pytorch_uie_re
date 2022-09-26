# pytorch_uie_re
基于pytorch的百度UIE关系抽取，源代码来源：[here](https://github.com/heiheiyoyo/uie_pytorch)。

百度UIE通用信息抽取的样例一般都是使用doccano标注数据，这里介绍如何使用通用数据集，利用UIE进行微调。

# 依赖

```
torch>=1.7.0
transformers==4.20.0
colorlog
colorama
```

# 步骤

- 1、数据放在data下面，比如已经放置的dgre（[工业知识图谱关系抽取-高端装备制造知识图谱自动化构建 竞赛 - DataFountain](https://www.datafountain.cn/competitions/584)），raw_data下面是原始的数据，新建一个process.py，将数据处理成类似mid_data下面的数据，```python process.py```即:

	```python
	{"id": "AT0004", "text": "617号汽车故障报告故障现象一辆吉利车装用MR479发动机，行驶里程为23709公里，驾驶员反映该车在行驶中无异响，但在起步和换挡过程中车身有抖动现象，并且听到离合器内部有异响。", "relations": [{"id": 0, "from_id": 0, "to_id": 1, "type": "部件故障"}, {"id": 1, "from_id": 2, "to_id": 3, "type": "部件故障"}], "entities": [{"id": 0, "start_offset": 80, "end_offset": 83, "label": "主体"}, {"id": 1, "start_offset": 86, "end_offset": 88, "label": "客体"}, {"id": 2, "start_offset": 68, "end_offset": 70, "label": "主体"}, {"id": 3, "start_offset": 71, "end_offset": 73, "label": "客体"}]}
	```

- 2、将mid_data下面的数据使用doccano.py转换成final_data下的数据，具体指令是：

	```python
	python doccano.py \
	    --doccano_file ./data/dgre/mid_data/doccano_train.json \
	    --task_type "ext" \  # ext表示抽取任务
	    --splits 0.9 0.1 0.0 \  # 训练、验证、测试数据的比例。训练，不对数据进行切分，因此将第一位设置为1.0
	    --save_dir ./data/dgre/final_data/ \
	    --negative_ratio 1  # 生成负样本的比率
	```
	
	最终会在final_data下生成train.txt、dev.txt。
	
- 3、将paddle版本的模型转换为pytorch版的模型：

	```python
	python convert.py --input_model=uie-base --output_model=uie_base_pytorch --no_validate_output
	```

	其中input_model可选的模型可参考convert.py里面。output_model是我们要保存的模型路径，下面会用到。之后我们可以测试下转换的效果：

	```python
	from uie_predictor import UIEPredictor
	from pprint import pprint
	
	schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
	ie = UIEPredictor('./uie_base_pytorch', schema=schema)
	pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
	```

- 4、开始微调：

	```python
	python finetune.py \
	    --train_path "./data/dgre/final_data/train.txt" \
	    --dev_path "./data/dgre/final_data/dev.txt" \
	    --save_dir "./checkpoint/dgre" \
	    --learning_rate 1e-5 \
	    --batch_size 8 \
	    --max_seq_len 512 \
	    --num_epochs 3 \
	    --model "uie_base_pytorch" \
	    --seed 1000 \
	    --logging_steps 464 \
	    --valid_steps 464 \
	    --device "gpu" \
	    --max_model_num 1
	```

	训练完成后，会在同目录下生成checkpoint/dgre/model_best/。

- 5、进行验证：

	```python
	python evaluate.py \
	    --model_path "./checkpoint/dgre/model_best" \
	    --test_path "./data/dgre/final_data/dev.txt" \
	    --batch_size 16 \
	    --max_seq_len 512
	```

- 6、使用训练好的模型进行预测：

	```python
	from uie_predictor import UIEPredictor
	from pprint import pprint
	
	schema = [{"主体":["部件故障", "性能故障", "检测工具", "组成"]}, "客体"] # Define the schema for entity extraction
	ie = UIEPredictor('./checkpoint/dgre/model_best', schema=schema)
	text = "分析诊断首先用故障诊断仪读取故障码为12，其含义是电控系统正常。考虑到电喷发动机控制是由怠速马达来实现的，所以先拆下怠速马达，发现其阀头上粘有大量胶质油污。用化油器清洗剂清洗后，装车试验，故障依旧。接着清洗喷油咀，故障仍未排除。最后把节气门体拆下来清洗。在操作过程中发现：一根插在节气门体下部真空管上的胶管已断裂，造成节气门后腔与大气相通，影响怠速运转稳定。这条胶管应该是连接在节气门进气管和气门室盖排气孔之间特制的丁字胶管的一部分，但该车没有使用特制的丁字胶管，它用一条直通胶管将节气门进气管和气门室盖排气孔连起来。维修方案把节气门体清洗干净后装车，再用一条专用特制的丁字形的三通胶管把节气门进气管、气门室盖排气孔和节气门体下部真空管接好，然后启动发动机，加速收油，发动机转速平稳下降"
	res = ie(text)
	pprint(res) # Better print results using pprint
	
	"""
	[{'主体': [{'end': 164,
	          'probability': 0.44656947,
	          'relations': {'性能故障': [{'end': 169,
	                                  'probability': 0.9079101,
	                                  'start': 167,
	                                  'text': '相通'}],
	                        '部件故障': [{'end': 169,
	                                  'probability': 0.91306436,
	                                  'start': 167,
	                                  'text': '相通'}]},
	          'start': 159,
	          'text': '节气门后腔'},
	         {'end': 153,
	          'probability': 0.92779523,
	          'relations': {'性能故障': [{'end': 156,
	                                  'probability': 0.9924409,
	                                  'start': 154,
	                                  'text': '断裂'}],
	                        '部件故障': [{'end': 156,
	                                  'probability': 0.9910217,
	                                  'start': 154,
	                                  'text': '断裂'}]},
	          'start': 151,
	          'text': '胶管'},
	         {'end': 68,
	          'probability': 0.32421115,
	          'relations': {'性能故障': [{'end': 77,
	                                  'probability': 0.6647082,
	                                  'start': 69,
	                                  'text': '粘有大量胶质油污'}],
	                        '部件故障': [{'end': 77,
	                                  'probability': 0.8483382,
	                                  'start': 69,
	                                  'text': '粘有大量胶质油污'}]},
	          'start': 66,
	          'text': '阀头'}],
	  '客体': [{'end': 77, 'probability': 0.5508156, 'start': 69, 'text': '粘有大量胶质油污'},
	         {'end': 156, 'probability': 0.9614242, 'start': 154, 'text': '断裂'},
	         {'end': 169,
	          'probability': 0.56304514,
	          'start': 164,
	          'text': '与大气相通'}]}]
	"""
	```

会发现，关系抽取会有关系的重复，可以多训练几个epoch看看。

# 补充

- 标签名最好是使用中文。
- 可使用不同大小的模型进行训练和推理，以达到精度和速度的平衡。
