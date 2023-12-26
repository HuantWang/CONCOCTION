import os,sys
import shutil
def append_file_contents(dynamic_file_path, composite_file_path):
    try:
        with open(dynamic_file_path, 'r') as dynamic_file:
            dynamic_content = dynamic_file.read()
        with open(composite_file_path, 'a') as composite_file:
            composite_file.write(dynamic_content)
        
        print(f"append {dynamic_file_path} to {composite_file_path}")
    
    except FileNotFoundError:
        print("FileNotFoundError")
    except Exception as e:
        print(f"error:{str(e)}")

def copy_files_with_same_name(static_dir, dynamic_dir, composit_dir):

    for root, _, files in os.walk(static_dir):
        for file_name in files:
            static_file_path = os.path.join(root, file_name)
            dynamic_file_path = os.path.join(dynamic_dir, file_name)
            composite_file_path = os.path.join(composit_dir, file_name)
            if os.path.exists(dynamic_file_path):
                shutil.copy(static_file_path, composite_file_path)
                append_file_contents(dynamic_file_path, composite_file_path)
                


if __name__ == "__main__":
    # static_dir = "/home/feature/dynamic/dynamic/jasper/result/static"
    # dynamic_dir = "/home/feature/dynamic/dynamic/jasper/result/dynamic"
    # composit_dir = "/home/feature/dynamic/dynamic/jasper/composite"
    static_dir=sys.argv[1]
    dynamic_dir=sys.argv[2]
    composit_dir=sys.argv[3]

    copy_files_with_same_name(static_dir, dynamic_dir, composit_dir)
