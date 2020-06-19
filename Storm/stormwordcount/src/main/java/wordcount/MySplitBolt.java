package wordcount;


import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;

import java.util.Map;

public class MySplitBolt extends BaseRichBolt {


    OutputCollector outputCollector;

    /*
     * 初始化初始化方法
     * */
    public void prepare(Map map, TopologyContext topologyContext, OutputCollector outputCollector) {
        this.outputCollector = outputCollector;
    }

    /*
     * 被storm 框架 while(true) 循环调用
     * */
    public void execute(Tuple tuple) {
        String line = tuple.getString(0);
        String[] words = line.split(" ");
        for (String word : words) {
            outputCollector.emit(new Values(word,1));
        }

    }

    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
        outputFieldsDeclarer.declare(new Fields("word","num"));
    }
}
