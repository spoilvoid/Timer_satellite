Tips: 所有脚本执行请在${root}文件夹（不出意外应该是Timer文件夹）完成，在测试容器时发现docker内部的python命令指向python2.7，故进行了scripts文件夹下相关脚本的更新，python->python3，后续解释器指向出问题可以看看是不是这个原因。
预训练数据处理部分：
    使用scripts/general_data_process_single_variable.sh脚本文件进行生成，需要在${root}/pretrain_data_gen/general_data文件夹下准备好对应的原始数据和params文件夹下面的data_tuple.json文件（list中每一个tuple都是一个需要处理的卫星·天）与X_range.json文件（单变量训练需要使用到所有的变量及其后处理时的阈值，请记录下所有的变量名以及想要处理成哪种类型，state、continous、multilabel处理完都变成了离散标签值，对单变量训练，如果没有特殊要求，请写为values类型且阈值可以任意填）。由于有效数据密度的筛选与n_delta设置的问题，可能导致部分特征拿不到数据，请根据具体情况进行动态调整，比如数据过度稀疏，可以尝试下跳数据密度阈值（甚至可以到0变成普通的数据处理）或者降低n_delta（这样满足最小时间点数量横跨的时间点就少了）。现在对所有的变量处理时都采用同一个数据密度阈值与n_delta，这里修改比较简单，如果实在有需求可以自己改一下
    随后使用mv ./pretrain_data_gen/general_data_processed/processed_data ./dataset/xw/pretrain/trainval放入对应的目录下（记得先行创建上级目录以防报错），为防止报错，也可以创建一个./dataset/xw/pretrain/test的空文件夹以应对test部分的数据载入（正常测试是不会报错的，保险起见这么操作一下）。
    现在的预训练处理脚本增加了multiprocessing的多进程并行部分，到时候根据cpu数量设置一下num_workers即可。
预训练部分：
    使用scripts/xw_pretrain_single_variable.sh进行模型单变量预训练，记得修改--nnodes --nproc_per_node --batch_size以适应新的机器的GPU与显存，这里虽然指定的是multivariate类型的数据，但实际也能够处理n_vars=1的数据，无需担心，记得放对数据路径，且如果是需要时间序列插补与时间序列异常检测等任务的预训练，请将--task_name分别对应改为imputation与anomaly_detection，由于只有预训练数据没有测试数据，请将train_test置为0，否则进程尝试进行测试集测试会报错。
微调部分：
    新增了--freeze_layer选项以冻结一部分模型层，既为了体现出预训练与微调的范式，也是可以提高微调的速度，只需要在原来的finetune脚本中增加--freeze_layer这个option即可，它会冻结模型的patch embedding层与projection层，其余参数不变，同时该操作只对Timer_multivariate生效（其余模型类没有实现但正常也不回去使用）。
