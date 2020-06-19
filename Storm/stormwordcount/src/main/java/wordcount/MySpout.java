package wordcount;

import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;

import java.util.Map;

public class MySpout extends BaseRichSpout {

    SpoutOutputCollector collector;

    /*
     * 初始化方法
     * */
    public void open(Map map, TopologyContext context, SpoutOutputCollector collector) {
        this.collector = collector;
    }

    /*
     * storm 框架在while(true),调用 nextTuple方法
     * */
    public void nextTuple() {
        collector.emit(new Values("Hello World I love China"));
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("firstStorm"));
    }
}