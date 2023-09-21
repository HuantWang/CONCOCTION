package cfg2path;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.util.HashMap;



class Node{
    public int id;
    public String attr;
    public List<Node> children;

    public Node() {}

    public Node(int _id,String _attr) {
        id = _id;
        attr=_attr;
        children=new ArrayList<>();
    }


}
class PreorderCfg{
    static HashMap<Integer, Node> nodes;
    static Node rootNode;
    static Node endNode;
    static String storedPath;
    static int MAX=50;
    static int deep=0;
    public  PreorderCfg(HashMap<Integer, Node> _nodes,String _storedPath){
        nodes=_nodes;
        storedPath=_storedPath;
        rootNode=findMinNode();
        System.out.println("这是最小id节点:"+rootNode.id);
        endNode=findMaxNode();
        System.out.println("这是最大id节点:"+endNode.id);
        //给cfg图的最后一个节点多加一个孩子节点(用于深度遍历到达叶子节点的标志)
        //给nodes的最后一个节点(即cfg图的最终节点)的孩子节点设为特定值id为-1,当遇到该节点时说明已经到达叶子节点.这样做的原理为cfg图第一个节点为method,最后尾节点为method_return
        Node e=new Node(-1,"this is the flag of the children node end");
        nodes.get(endNode.id).children.add(e);
    }
    public static List<Node> preorder() {
        List<Node> res = new ArrayList<>();
        helper(rootNode, res);
        return res;
    }

    public static void helper(Node root, List<Node> res) {
        if (root.id==-1) {
            System.out.println("到叶子节点了!");
            writeFile(storedPath,res);
            return;
        }
        res.add(root);
        for (Node ch : root.children) {
            boolean isVisited=false;
            for(Node n:res){
                if(n.id==ch.id){
                    isVisited=true;
                    for(Node nn:res) System.out.print(nn.id+" ");
                    System.out.println("end");
                    break;
                }

            }
            if(isVisited)continue;
            helper(ch, res);
            res.remove(ch);




        }
    }



    public static void run(){
        List<Node>path=new ArrayList<>();
        path.add(rootNode);
        dfs(rootNode,path,endNode);
    }
    public static void dfs(Node root,List<Node> path,Node target){
        deep++;
        if(deep>MAX){
            deep--;
            writeFile(storedPath,path);
            return;
        }

        if(root==target){
            writeFile(storedPath,path);
//            for(Node n:path)
//            System.out.print(n.id+",");
//            System.out.println("-----------");

        }else{
            for(Node n :root.children){
                Node next=n;
                List <Node> tpath=new ArrayList<>(path);//复制list 使二者存储在不同地址空间
               // List <Node> tpath=path;
                boolean inThepath=false;//路径当中已有此节点 此时访问说明有环
                for(Node t:path){
                    if(t.id==next.id){inThepath=true;break;}
                }
                if(!inThepath){
                    tpath.add(next);
                    dfs(next,tpath,target);
                }else{

//                    String s="";
//                    for(Node o:tpath)s=s+o.id+",";
//                    s=s+next.id;
//                    System.out.println(s);

                    //此时next节点已访问过,若是回路的起点节点(即:有孩子节点已访问过,有孩子节点没有访问过),则继续递归访问该节点
                    //此时next节点已访问过,若过所有的孩子节点都已经访问过那么,不递归访问该节点
                    boolean isCircleStart=false;
                    for(Node t:next.children){
                       if(!path.contains(t)){
                          isCircleStart=true;
                       }
                    }
                    if(isCircleStart){tpath.add(next);dfs(next,tpath,target);}


                }
            }
        }
        deep--;

    }


    static void writeFile(String path,List<Node> res){
        BufferedWriter writer;
        String s = "";
        try{
            writer=new BufferedWriter(new FileWriter(path,true));
            if(res.isEmpty())
                return;
            for(Node n:res){
                s=s+n.id+",";
            }
            s=s.substring(0,s.length()-1);
            System.out.println("path:"+s);
            writer.write(s+"\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    //按照节点id大小对hashmap排序 递增
    static Node findMinNode(){
        Set set = nodes.keySet();
        Object[] arr = set.toArray();
        Arrays.sort(arr);
//        for(int i=0;i< arr.length;i++)
//            System.out.println(arr[i]);
 //       System.out.println("====================================");
        return nodes.get(arr[0]);
    }
    static Node findMaxNode(){
        Set set = nodes.keySet();
        Object[] arr = set.toArray();
        Arrays.sort(arr);
        return nodes.get(arr[arr.length-1]);
    }

    public static Node getRootNode() {
        return rootNode;
    }
}

public class GetCfgInfo {

    static Pattern nodeId=Pattern.compile("\"[0-9]*\"");
    static Pattern nodeAttr=Pattern.compile("\\[.*\\]");
    static Pattern edge=Pattern.compile(" -> ");
 //   static String storedPath="E:\\tt\\badd_result\\result.txt";




    public static void main(String[] args){


        String fileDir=args[0]; //包含cfg图信息文本以及ast节点源码信息文本 的文件夹
        String storedDir=args[1];//存储处理后cfg特征的文件夹
//        String fileDir="E:\\tt\\test_getcfg";
//        String storedDir="E:\\tt\\test_cfgR";

        File f=new File(storedDir);
        if(!f.exists()){
            f.mkdir();
        }else{
            f.delete();
            f.mkdir();
        }

        File folder = new File(fileDir);
        String fileList[] = folder.list();
        Arrays.sort(fileList);
        List<String> cfgList=new ArrayList<>();
        List<String> astList=new ArrayList<>();


        for (int i = 0; i < fileList.length; i++) { // 遍历返回的字符数组
            String s= String.valueOf(fileList[i]);
            if(s.endsWith("cfg.txt"))
            cfgList.add(fileDir+"/"+s);
            if(s.endsWith("ast.txt"))
                astList.add(fileDir+"/"+s);
        }
        for(int i=0;i< astList.size();i++){
            String cfgFile=cfgList.get(i);
            String astFile=astList.get(i);
            String filenema=astFile.substring(astFile.lastIndexOf("/"),astFile.lastIndexOf("-ast.txt"));
            String storedPath=storedDir+filenema+"-cfgResult.txt";
            String s1=cfgFile.split("cfg.txt")[0];
            String s2=astFile.split("ast.txt")[0];
            if(s1.equals(s2)){
                run(cfgFile,astFile,storedPath);
                System.out.println("stored in : "+storedPath);
                System.out.println("============================================================================================================");
            }
        }


//        run("E:\\tt\\bad_rawResult_new\\CWE416_Use_After_Free__malloc_free_char_17.c-CWE416_Use_After_Free__malloc_free_char_17_bad.c-cfg.txt",
//                "E:\\tt\\bad_rawResult_new\\CWE416_Use_After_Free__malloc_free_char_17.c-CWE416_Use_After_Free__malloc_free_char_17_bad.c-ast.txt",
//                "E:\\result.txt");



    }
    static void run(String cfgFilePath,String astFilePath,String storedPath){
        String funcName=astFilePath.split(".c-")[1];
        System.out.println("this is stored path: "+storedPath);

        Node2Code n2c = new Node2Code();

        //判断文件存在则删除文件
        File file = new File(storedPath);
        if(file.exists()) file.delete();



        //通过文本解析,获得基于Node数据结构下的cfg表示
        HashMap<Integer, Node> nodes=parseTxt2node(cfgFilePath,funcName);
        if(nodes.size()==0){
            System.out.println("Error : cfg is null:nodes.size()==0!");return;
        }
        //根据nodes得出所有边写入文件
        writeCfg(nodes,storedPath);
        //深度遍历,并将每一条独立路径写入文件
        PreorderCfg preorderCfg=new PreorderCfg(nodes,storedPath);
       // preorderCfg.preorder();
        preorderCfg.run();
       // System.out.println("======================================================");
        System.out.println("cfg and Independent path are stored in : "+storedPath);

        //将ast节点源码与节点id相匹配 并写入文件
        //减2 因为ast文件中第三个节点才是cfg中序号最小的节点
        Node rootNode=preorderCfg.getRootNode();
        n2c.node2code(astFilePath,storedPath, rootNode.id-2);
    }
    static void writeCfg(HashMap<Integer, Node> nodes, String path){
        BufferedWriter writer;
        try {
            writer=new BufferedWriter(new FileWriter(path,true ));
            writer.write("-----cfg-----\n");
            for(int i :nodes.keySet()){
                Node node=nodes.get(i);
                for(Node n:node.children){
                    if(n.id!=-1){
                        String s="("+node.id+","+n.id+")\n";
                        writer.write(s);
                    }
                }

            }
            writer.write("-----path-----\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    static HashMap<Integer, Node>  parseTxt2node(String filePath,String funcName){

        BufferedReader reader;
        HashMap<Integer, Node> nodes = new HashMap<Integer, Node>();
        try {//按行读入文件,对每一行进行处理:存储节点或者存储边
            reader = new BufferedReader(new FileReader(filePath));
            String line = reader.readLine();
            boolean flag=false;//是否是目标函数的cfg信息
            while (line != null) {
                if(line.contains(funcName))flag=true;
                if(flag==false){line = reader.readLine();continue;}
                Matcher id=nodeId.matcher(line);
                Matcher attri=nodeAttr.matcher(line);
                Matcher edgee=edge.matcher(line);
                if (id.find() &&attri.find()) {
                    //读入节点以及节点信息
                    Node tempNode=new Node(Integer.parseInt(id.group(0).replace("\"","")),attri.group(0));
                    nodes.put(Integer.parseInt(id.group(0).replace("\"","")),tempNode);}
                else if(edgee.find()){
                    //读入边信息 12 -> 13 在起点节点的chdren中增加子节点 13
                    int flag1=edgee.start();
                    int flag2=edgee.end();
                    String start=line.substring(1,flag1).replace("\"","").replace(" ","");
                    String end=line.substring(flag2).replace("\"","").replace(" ","");
                    nodes.get(Integer.parseInt(start)).children.add(nodes.get(Integer.parseInt(end)));
                    }

                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i : nodes.keySet()){
            System.out.println("node id: "+i+" value: "+nodes.get(i));
        }
       return nodes;
    }

    //解析cfg.txt 将点和源码信息提取对应
    static void parseCfgCode(Node node){
        Pattern nodeCode1=Pattern.compile(";\\.\\w*,.*\\)<");//匹配格式为 ;.操作符,源码)< 格式
        Pattern nodeCode2=Pattern.compile(";\\.\\w*,.*\\)<");//匹配格式为 ;.操作符,源码)< 格式
    }









}
