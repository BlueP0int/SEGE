洗钱数据集生成步骤：
1. moneyLunderingGraphEncoder.py，将实际洗钱数据构建成一个多边的有向图，对不同字段进行编码
	输入：真实洗钱账户excel表格
	输入：moneyLunderingGraphGroundTruth.gml，是一个nx.MultiDiGraph()，存在平行边的有向图
	输出示例：
	  edge [
		source 419
		target 315
		key 25
		datetime "20131105000000"
		weight 673432.0
		feature "[1.0, 0.0, 0.0, 1.0,……]
		]
2. GTMultiGraphCombine.py，将存在平行边的MultiDiGraph合并成为无平行边的DiGraph
	输入：moneyLunderingGraphGroundTruth.gml
	输出：moneyLunderingDiGraphCombined.gml
	
3. generate_weighted_sbm.py，根据GroundTruth洗钱数据构建完整数据集。
	输入：moneyLunderingDiGraphCombined.gml
	输出：combined_20220729_lower300000.gml和groundTruth部分对应的节点label gt_20220729_lower300000.txt
	分为3步：生成sbm无权图的代码-->生成sbm带权图的代码--> 将groundtruth中入度或出度为0的点嵌入sbm模型的代码，生成最终的图
