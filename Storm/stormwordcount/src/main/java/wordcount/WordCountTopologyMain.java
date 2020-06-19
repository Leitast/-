package wordcount;
import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.generated.AlreadyAliveException;
import org.apache.storm.generated.InvalidTopologyException;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.tuple.Fields;

public class WordCountTopologyMain {

    public static void main(String[] args) throws AlreadyAliveException, InvalidTopologyException {


        // 1、准备一个 TopologyBuilder
        TopologyBuilder topologyBuilder = new TopologyBuilder();
        topologyBuilder.setSpout("mySpout",new MySpout(),1);
        topologyBuilder.setBolt("mybolt1",new MySplitBolt(),10).shuffleGrouping("mySpout");
        topologyBuilder.setBolt("mybolt2",new MyCountBolt(),2).fieldsGrouping("mybolt1",new Fields("word"));

        //2、创建configuration,指定topology 需要的worker的数量
        Config config=new Config();
        config.setNumWorkers(2);

        //3、提交任务，分为本地模式、集群模式
        // StormSubmitter.submitTopologyWithProgressBar("mywordcount",config,topologyBuilder.createTopology());

        LocalCluster localCluster = new LocalCluster();
        localCluster.submitTopology("mywordcount",config,topologyBuilder.createTopology());
    }
}