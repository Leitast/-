package CUG.SE.lEITAST;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.ipc.Server;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HBaseTest {
    public static Connection connection;
    public static Admin admin;
    public static Configuration configuration;


    public static void connect() throws IOException {
        configuration = HBaseConfiguration.create();
        configuration.set("hbase.zookeeper.quorum","");   //添加你的ip地址
//        configuration.set("hbase.zookeeper.property.clientPort","2181");
        try{
            connection = ConnectionFactory.createConnection(configuration);
            admin = connection.getAdmin();
        }catch (IOException e){
            e.printStackTrace();
        }
    }
    public static void main(String[] args) throws IOException {
        //读取csv文件
        List<String> dataList = new ArrayList<String>();//数据
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new FileReader("D:/workspace/ideaworkspace/HBase/src/20200202.export.csv"));//为文件流赋数据
            //bufferedReader = new BufferedReader(new FileReader("/home/zjj/桌面/test.csv"));//为文件流赋数据
            String line = "";
            while ((line = bufferedReader.readLine())!=null){
                dataList.add(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            if(bufferedReader!=null){
                try {
                    bufferedReader.close();
                    bufferedReader = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println(dataList.size());
        connect();
        Table table = connection.getTable(TableName.valueOf("hbase_test"));
        for(int i=0;i<dataList.size();i++)
        {
            //获取当前列数据
            String dataline=dataList.get(i);
            //用逗号分割
            String []data =dataline.split(",");

            for(int j=0;j<58;j++)
            {
                System.out.print(data[j]);
                Put put = new Put( Integer.toString(i).getBytes());
                put.addColumn("data".getBytes(), Integer.toString(j).getBytes(), data[j].getBytes());
            }
            System.out.println(0);
        }
        table.close();
        System.out.println("insert successfully");
    }

    public static void close(){
        try{
            if(admin != null){
                admin.close();
            }
            if(null != connection){
                connection.close();
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public static void listTables() throws IOException {
        connect();
        HTableDescriptor hTableDescriptors[] = admin.listTables();
        for(HTableDescriptor hTableDescriptor :hTableDescriptors){
            System.out.println(hTableDescriptor.getNameAsString());
        }
        close();
    }



}
