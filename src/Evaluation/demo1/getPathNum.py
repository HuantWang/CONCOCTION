import sys
def count_lines_with_keywords(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            count = 0
            flag='null'
            for line in lines:
                if '-----cfgNode-----' in line:
                    flag='null'
                    break
                if(flag=='path'):
                    count += 1
                elif '-----path-----' in line:
                   flag="path" 
            return count
    else:
        return 0
    
if __name__=='__main__':
    file=sys.argv[1]
    print(count_lines_with_keywords(file))