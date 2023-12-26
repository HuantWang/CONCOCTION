package cfg2path;

import java.io.*;

//将原来的静态特征和新特征添一起
public class Concate2File {
    public static void main(String[] args) {
//        String cfgDir=args[0]; //cfg特征文件夹
//        String stroePathDir=args[1];//seven edges 特征文件夹
        String cfgDir=args[0]; //cfg特征文件夹
        String stroePathDir=args[1];//seven edges 特征文件夹
        File cfgFolder = new File(cfgDir);
        String cfgfileList[] = cfgFolder.list();
//        File storedFolder = new File(stroePathDir);
//        String storedList[] = storedFolder.list();
        for(String s:cfgfileList){
            String sevenEdgepath=stroePathDir+"/"+s.substring(0,s.lastIndexOf("-cfgResult.txt"))+".txt";
            String cfgpath=cfgDir+"/"+s;
            String temp=stroePathDir+"/"+s.substring(0,s.lastIndexOf("-cfgResult.txt"))+"-temp.txt";

            BufferedReader reader0;
            BufferedReader reader;
            BufferedWriter writer;

            try {//按行读入文件,对每一行进行处理:存储节点或者存储边
                reader0 = new BufferedReader(new FileReader(sevenEdgepath));
                reader = new BufferedReader(new FileReader(cfgpath));
                writer=new BufferedWriter(new FileWriter(temp,true));
                //先将7条边特征写入temp 去除最后一行标志
                String line0 = reader0.readLine();
                while (line0 != null) {
                    if(line0.contains("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")){
                        break;//到达最后一行 退出 不写入temp
                    }

                    writer.write(line0+"\n");
                    line0 = reader0.readLine();
                }
                reader0.close();
                //将cfg信息写入temp
                String line = reader.readLine();
                while (line != null) {
                    writer.write(line+"\n");
                    line = reader.readLine();
                }
                reader.close();
                writer.close();
            } catch (IOException e) {e.printStackTrace();}

            File old=new File(sevenEdgepath);
            old.delete();
            File now=new File(temp);
            File rneme=new File(sevenEdgepath);
            now.renameTo(rneme);
            System.out.println("writing : "+sevenEdgepath);

        }









    }
}
