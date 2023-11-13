from bs4 import BeautifulSoup
import os


def GetText_HTML(file):
    html_content = ''
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    text = "\n".join(soup.get_text().split('\n')[6:])
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text


def Convert_HTML_TEXT(read_dir, target_dir):
    document_id = 1
    for directories in os.listdir(read_dir):

        # read the 1 parent directory
        directory_path = os.path.join(read_dir, directories)
        # create same diretory to sore converted file
        write_parent_directory_path = os.path.join(target_dir, directories)
        os.makedirs(write_parent_directory_path, exist_ok=True) 

        for directory in os.listdir(directory_path):

            # read path for sub directory
            sub_directories = os.path.join(directory_path, directory)
            # write path for sub directory
            write_sub_directory_path = os.path.join(write_parent_directory_path, directory)
            os.makedirs(write_sub_directory_path, exist_ok=True)

            for sub_directory in os.listdir(sub_directories):

                print(f"Converting File {sub_directory}")
                sub_directory_path = os.path.join(f"{sub_directories}\{sub_directory}")
                text = GetText_HTML(sub_directory_path)
                write_file_path = os.path.join(write_sub_directory_path, str(document_id))
                with open(write_file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    document_id += 1



if __name__ == "__main__":
    read_path = os.getcwd() + "\webkb"
    write_path = "D:\My Work\Final Year Project\Git\FYP23-Deep-Document-Clustering\Base Lines\Lexical Chains\WebKB\\"
    Convert_HTML_TEXT(read_path, write_path)
    

