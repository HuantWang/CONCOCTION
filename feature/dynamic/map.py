import os
import sys
from shutil import copyfile

####################
# main.py paths文件 dynamic文件夹路径 匹配的dynamic输出路径
# 目前sard的dynamic中没有if语句
# dynamic空  1CWE416_Use_After_Free__malloc_free_char_01.c-CWE416_Use_After_Free__malloc_free_char_01_good.c

#      /\_/\
#    ( o   o )
#   =(  =^=  )=
#     (     )
#      V---V

#找不到匹配 打造CWE416_Use_After_Free__malloc_free_int_12.c-goodG2B.c
####################

# 得到参数

# paths_path = sys.argv[1]
# dynamic_path = sys.argv[2]
def map_dynamic(paths_path,dynamic_path,new_path):
    # 获取动态文件
    c=0
    start = "--------------path--------------"
    end = "--------------path over--------------"

    paths_filename_list = os.listdir(paths_path)
    paths_filename_list.sort()

    import shutil
    # 确认输出目录
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        os.makedirs(new_path)
    else:
        os.makedirs(new_path)

    for path_filename in paths_filename_list:
        paths_filepath=paths_path+"/"+path_filename
        # 读paths内容
        path_k = 0
        flag = 0
        start_num = []
        end_num = []
        with open(paths_filepath, "r") as f_paths:
            paths_line = f_paths.readlines()
        # new_out_path = new_path + "/" + path_filename.strip(".c.txt")
        # os.makedirs(new_out_path)
        new_out_path = new_path

        # print(len(paths_line))
        # 找到所有的paths——k
        if len(paths_line) > 0:
            for i, p_line in enumerate(paths_line, start=1):
                if p_line.strip() == start:
                    flag = 1
                    start_num.append(i)
                    path_k = path_k + 1
                if (p_line.strip() == end) & (flag == 1):
                    flag = 0
                    end_num.append(i - 1)
                #if p_line.strip() == "--------------code--------------":
                    #name_pathfile= paths_line[i].split(" ")[-1]
                    #name_pathfile = paths_line[i].split(" ")[-1].replace('(', '').replace(')', '')
                    #print(name_pathfile)

        # print(start_num)
        # print(end_num)
        # print(paths_line[end_num[0]])
        # 依次读取每个文件

        for root, dirs, files in os.walk(dynamic_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        #print(file_path)
                        dynamic_filename=file
                        print(file)
                        #name_dynamic = dynamic_filename.split("-")[-1]
                        #name_dynamicfile = name_dynamic.split(".")[0]
                        name_dynamicfile = dynamic_filename.lstrip('0123456789')
                        # print(name_dynamicfile.strip())
                        # print(path_filename.strip())
                        if name_dynamicfile.strip() != path_filename.strip():
                            continue
                        c=c+1


                        # print(dynamic_filename)
                        start_paths = 0
                        end_paths = 0
                        # \\->\
                        with open( file_path, "r") as f_dynamic:
                            dynamic_line = f_dynamic.readlines()

                            # 找到开始位置和结束位置
                            for i, d_line in enumerate(dynamic_line, start=1):
                                if d_line.strip() == r"=========trace=========":
                                    start_dynamic = i
                                    dynamic_line[i]=dynamic_line[i].replace('{','').strip()
                                    #print(dynamic_line[i])
                                if d_line.strip() == r"=======================":
                                    end_dynamic = i - 1
                                    # print(dynamic_line[i -1])
                            # 开始匹配
                            # 防止没有dynamic
                            if start_dynamic == end_dynamic:
                                continue
                            flag_mate = 0
                            # 与所有的paths进行匹配
                            for i in range(path_k):
                                # 取出第i个
                                start_paths = start_num[i]
                                end_paths = end_num[i]

                                num_paths = start_paths
                                num_dynamic = start_dynamic
                                while (1):
                                    # print(num_paths,num_dynamic)
                                    #匹配括号 if fprintf for while
                                    if paths_line[num_paths].lstrip()[:7] == "fprintf":
                                        #paths_line[num_paths] = paths_line[num_paths].strip().lstrip("{")
                                        print(len(paths_line[num_paths]))
                                        while (1):
                                            left_num = 0
                                            right_num = 0
                                            for i in range(len(paths_line[num_paths])):
                                                if paths_line[num_paths][i] == "(":
                                                    left_num = left_num + 1
                                                elif paths_line[num_paths][i] == ")":
                                                    right_num = right_num + 1
                                            if left_num != right_num:
                                                if num_paths+1 == end_paths:
                                                    break
                                                paths_line[num_paths + 1] = paths_line[num_paths] + paths_line[ num_paths + 1]
                                                num_paths = num_paths + 1
                                            else:
                                                break

                                    #for
                                    if paths_line[num_paths].lstrip()[:3] == "for":
                                        #paths_line[num_paths] = paths_line[num_paths].strip().lstrip("{")
                                        print(len(paths_line[num_paths]))
                                        while (1):
                                            left_num = 0
                                            right_num = 0
                                            for i in range(len(paths_line[num_paths])):
                                                if paths_line[num_paths][i] == "(":
                                                    left_num = left_num + 1
                                                elif paths_line[num_paths][i] == ")":
                                                    right_num = right_num + 1
                                            if left_num != right_num:
                                                if num_paths+1 == end_paths:
                                                    break
                                                paths_line[num_paths + 1] = paths_line[num_paths] + paths_line[ num_paths + 1]
                                                num_paths = num_paths + 1
                                            else:
                                                break
                                    # if
                                    if paths_line[num_paths].lstrip()[:2] == "if":
                                        #paths_line[num_paths] = paths_line[num_paths].strip().lstrip("{")
                                        print(len(paths_line[num_paths]))
                                        while (1):
                                            left_num = 0
                                            right_num = 0
                                            for i in range(len(paths_line[num_paths])):
                                                if paths_line[num_paths][i] == "(":
                                                    left_num = left_num + 1
                                                elif paths_line[num_paths][i] == ")":
                                                    right_num = right_num + 1
                                            if left_num != right_num:
                                                if num_paths + 1 == end_paths:
                                                    break
                                                paths_line[num_paths + 1] = paths_line[num_paths] + paths_line[num_paths + 1]
                                                num_paths = num_paths + 1
                                            else:
                                                break
                                    #while
                                    if paths_line[num_paths].lstrip()[:5] == "while":
                                        #paths_line[num_paths] = paths_line[num_paths].strip().lstrip("{")
                                        print(len(paths_line[num_paths]))
                                        while (1):
                                            left_num = 0
                                            right_num = 0
                                            for i in range(len(paths_line[num_paths])):
                                                if paths_line[num_paths][i] == "(":
                                                    left_num = left_num + 1
                                                elif paths_line[num_paths][i] == ")":
                                                    right_num = right_num + 1
                                            if left_num != right_num:
                                                if num_paths+1 == end_paths:
                                                    break
                                                paths_line[num_paths + 1] = paths_line[num_paths] + paths_line[ num_paths + 1]
                                                num_paths = num_paths + 1
                                            else:
                                                break
                                    print(paths_line[num_paths].strip().rstrip("{"))
                                    print(dynamic_line[num_dynamic].strip().rstrip("{"))
                                    if paths_line[num_paths].strip().rstrip("{").replace(" ","") == dynamic_line[num_dynamic].strip().rstrip("{").replace(" ",""):
                                        # 语句能够匹配，dynamic和path下一句
                                        num_paths = num_paths + 1
                                        num_dynamic = num_dynamic + 1
                                    else:
                                        # 语句不匹配，dynamic下一句
                                        num_dynamic = num_dynamic + 1
                                    # windows \\ linux \

                                    if (num_paths == end_paths):
                                        # path走完，能完全匹配
                                        copyfile( file_path, new_out_path+"/"+ dynamic_filename)

                                        flag_mate = 1
                                        break
                                    elif (num_dynamic == end_dynamic):
                                        # dynamic走完
                                        if num_dynamic - start_dynamic == num_paths - start_paths:
                                            # dynamic过短，dynamic所有内容都在path中

                                            copyfile( file_path, new_out_path+"/"+ dynamic_filename)
                                            flag_mate = 1
                                            break
                                        else:
                                            # 未能匹配成功
                                            break
                                if flag_mate == 1:
                                    break
        # print("匹配上:")
        # print(c)
# map_dynamic(paths_path,dynamic_path,new_path)
def main():
    paths_path = sys.argv[1]
    dynamic_path =sys.argv[2]
    new_path =sys.argv[3]
    map_dynamic(paths_path, dynamic_path, new_path)

if __name__== "__main__" :
    main()
