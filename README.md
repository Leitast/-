# Big-Data
CUG-2020-软工选修课《大数据技术与应用》课内作业，包括HBase、Storm和Sklearn的使用

本研究基于GDELT新闻数据进行实验,如有需要,请自行下载相关数据。

## HBase
* 启动路径：`ip`:16010/
* 连接云服务器的HBase
* 向云服务器的HBase中插入数据‘
* 返回HBase中所有的Table名<br>

## Storm
* 启动路径：`ip`:8080/
* 设置需要进行词频统计的句子
* 进行package操作，将jar部署到云服务器上
* 执行$ storm jar `你的jar包名字`.jar wordcount.WordCountTopologyMain<br>

## Sklearn(随机森林回归模型)
* 选择了GDELT数据中第30，31，32，33，34，35字段数据(其余字段有缺失)
* 通过潜在影响、新闻次数、数据源、文章数和语气等5个属性指标对事件类型进行预测
* `GDELT_Analysis.py`是用来从GDELT的数据中提取字段的
* `RFRegression.py`是随机森林回归模型的主要代码
可通过更改`RFRegression.py`中的相关参数应用到其余适合随机森林研究的任何场景
