package wordcount;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.IBasicBolt;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.IWindowedBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Tuple;

import java.util.HashMap;
import java.util.Map;

public class MyCountBolt extends BaseRichBolt {


    OutputCollector collector;
    Map<String, Integer> map = new HashMap<String, Integer>();

    public void prepare(Map map, TopologyContext topologyContext, OutputCollector outputCollector) {
        this.collector = outputCollector;
    }

    public void execute(Tuple tuple) {
        String word = tuple.getString(0);
        Integer num = tuple.getInteger(1);

        if (map.containsKey(word)) {
            Integer count = map.get(word);
            map.put(word, count + num);
        }else{
            map.put(word,1);
        }

        System.out.println("count:"+map);

    }

    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {

    }
}