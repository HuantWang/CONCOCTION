package cfg2path;

import java.io.*;
import java.util.HashMap;
import java.util.regex.Matcher;

public class Node2Code {

    public  void node2code(String astPath, String stroePath, int nodeStartId) {
//        String astPath="E:\\tt\\t\\tifftool_ast.txt";
//        String stroePath="E:/tt/t/result2.txt";
//        int nodeStartId=7;
        BufferedReader reader;
        BufferedWriter writer;

        try {//按行读入文件,对每一行进行处理:存储节点或者存储边
            reader = new BufferedReader(new FileReader(astPath));
            writer=new BufferedWriter(new FileWriter(stroePath,true));
            writer.write("-----cfgNode-----\n");
            String line = reader.readLine();
            while (line != null) {
                String s=nodeStartId+",";
                s=s+line+"\n";
                writer.write(s);
                line = reader.readLine();
                nodeStartId++;
            }
            reader.close();
            writer.write("=====================================");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
       // System.out.println("======================================================");
        System.out.println("the code of the node from cfg are stored in : "+stroePath);




    }
}
