import re
import os
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被automated_good.py调用执行
#对整个trace进行函数级的切片

folder_name = sys.argv[1]
path = sys.argv[2]
your_path = path + "/testcase-trace/" + folder_name + "/"
# your_path = r"/xxx/yyy/replay/testcase-trace/"+folder_name+"/"
if (os.path.exists(your_path) == False):
    os.makedirs(your_path)
regex1 = r"(?<=file_name:).*"
regex2 = r"(?<=function_name:).*"

stack = []
start_line_number = None
with open(path + '/cut/' + folder_name + '.txt', 'r') as f:
    lines = f.readlines()
    for line_number, line_contents in enumerate(lines, start=1):
        if line_contents.strip() == r"------function start!------":
            stack.append(line_number)
        elif line_contents.strip() == r"------function end!------":
            if len(stack) > 0:
                start_line_number = stack.pop()
                local_filename = lines[start_line_number - 3].split(':')[-1]
                local_function = lines[start_line_number - 2].split(':')[-1].split('/')[-1].strip('\n')
                filename = local_filename.split('/')[-1]
                filename.strip('\n')
                print(lines[start_line_number - 3])
                print(filename.strip())
                print(local_function.strip())
                str = filename.strip()+ "-" + local_function.strip() + ".c.txt"
                file = your_path + str
                with open(f'{file}', 'w', encoding='utf-8') as output_file:
                    output_file.write("----------------dynamic----------------\n")
                    output_file.write("=======testcase========\n")
                    output_file.write("=========trace=========\n")
                    output_file.writelines(lines[start_line_number:line_number - 1])
                    for delete_line_number in range(start_line_number - 3, line_number):
                        lines[delete_line_number] = "\n"
                    output_file.write("=======================\n")
                    output_file.close()
                str_new = filename[10:] + r"-" + local_function[14:] + r".c_new.txt"
                file_new = your_path + str_new
                with open(f'{file}', 'r', encoding='utf-8') as output_file_delete:
                    lines_delete = output_file_delete.readlines()
                    with open(f'{file_new}', 'a+', encoding='utf-8') as output_file_new:
                        for line_number_dellete, line_contents_delete in enumerate(lines_delete, start=1):
                            if line_contents_delete != "\n":
                                output_file_new.write(line_contents_delete)
                os.remove(file)
                os.rename(file_new, file)
        elif line_contents.strip() == r"=======================":
            while len(stack) > 0:
                start_line_number = stack.pop()
                local_filename = lines[start_line_number - 3]
                local_function = lines[start_line_number - 2]
                filename = os.path.split(local_filename)[1]
                filename.split('\n')
                str = filename[10:-1] + "-" + local_function[14:-1] + ".c.txt"
                file = your_path + str
                with open(f'{file}', 'w', encoding='utf-8') as output_file:
                    output_file.write("----------------dynamic----------------\n")
                    output_file.write("=======testcase========\n")
                    output_file.write("=========trace=========\n")
                    output_file.writelines(lines[start_line_number:line_number - 1])
                    for delete_line_number in range(start_line_number - 3, line_number-1):
                        lines[delete_line_number] = "\n"
                    output_file.write("=======================\n")
                    output_file.close()
                str_new = filename[10:] + r"-" + local_function[14:] + r".c_new.txt"
                file_new = your_path + str_new
                with open(f'{file}', 'r', encoding='utf-8') as output_file_delete:
                    lines_delete = output_file_delete.readlines()
                    with open(f'{file_new}', 'a+', encoding='utf-8') as output_file_new:
                        for line_number_dellete, line_contents_delete in enumerate(lines_delete, start=1):
                            if line_contents_delete != "\n":
                                output_file_new.write(line_contents_delete)
                os.remove(file)
                os.rename(file_new, file)