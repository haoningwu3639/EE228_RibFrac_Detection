# batch_rename.py
# Created: 18th March 2021

"""
You can use a parameter 'mode' to choose to change file suffix in batches
or to change the filename into a clean and orderly format, for example, 00001.png
"""

__author__ = "HaoningWu"
__version__ = "1.0"

import os
import argparse

def change_suffix(input_dir, old_suffix, new_suffix):
    fileList = os.listdir(input_dir)
    for fileName in fileList:
        root_name, file_suffix = os.path.splitext(fileName)
        if file_suffix == old_suffix:
            NewFile = root_name + new_suffix
            # Replace the old filename with a new one
            os.rename(os.path.join(input_dir, fileName), os.path.join(input_dir, NewFile))
    print("Batch Change Suffix Finished!")
    print(os.listdir(input_dir))

def batch_rename(input_dir, length, start_num, pattern, sort_key, sort_order):
    fileList = os.listdir(input_dir)
    order = False if sort_order == 'ascend' else True
    # os.path.getmtime()    obtain the last modified time
    # os.path.getctime()    obtain the last created time
    if sort_key == 'mtime' or 'ctime':
        fileList = sorted(fileList, key = lambda x:os.path.getmtime(os.path.join(input_dir, x)), reverse = order) # sorted True for Descending and False for Ascending
    else:
        fileList = sorted(fileList, reverse=order)
    os.chdir(input_dir)  # Move to the working directory
    for fileName in fileList:
        if os.path.splitext(fileName)[-1] == pattern:
            
            os.rename(fileName, fileName[:10] + '-label' + fileName[15:])
            start_num += 1
        else:
            continue

    print("Batch Rename Finished!")
    print(os.listdir(input_dir))


def get_parser():
    # --input_dir means it's nonessential, while input_dir means it's essential
    parser = argparse.ArgumentParser(description='batch rename files in a directory')
    # Please use absolute path here
    parser.add_argument('--input_dir', default=r"C:\Users\Administrator\Documents\GitHub\PythonScripts\data\\", type=str, help='the directory where to rename')
    # The mode can be either 'filename' or 'suffix'
    parser.add_argument('--mode', default='filename', type=str, help='choose to change the file name or the suffix of the file')
    # arguments for change_suffix
    parser.add_argument('--old_suffix', default='.png', type=str, help='old version of the suffix')
    parser.add_argument('--new_suffix', default='.jpg', type=str, help='new version of the suffix')
    # arguments for batch_rename
    parser.add_argument('--length', default=5, type=int, help='the length of the filename, for example, 00001.suffix when you use the default choice')
    parser.add_argument('--start_num', default=1, type=int, help='the number of the first filename')
    parser.add_argument('--pattern', default='.png', type=str, help='the pattern of the file you want to batch rename')
    # The sort_key can be name, mtime, ctime; sort_order can be ascend or descend
    parser.add_argument('--sort_key', default='name', type=str, help='you can choose sort and rename files according to the filename/modeified time/created time')
    parser.add_argument('--sort_order', default='ascend', type=str, help='you can choose rename in ascending/descending order')
    
    return parser


def main():
    # Add command line arguments
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    mode = args.mode
    old_suffix = args.old_suffix
    new_suffix = args.new_suffix
    length = args.length
    start_num = args.start_num
    pattern = args.pattern
    sort_key = args.sort_key
    sort_order = args.sort_order

    # Complete suffix
    if old_suffix and old_suffix[0] != '.':
        old_suffix = '.' + old_suffix
    if new_suffix and new_suffix[0] != '.':
        new_suffix = '.' + new_suffix    
    if pattern and pattern[0] != '.':
        pattern = '.' + pattern
    # Choose the rename mode
    if mode == 'filename':
        batch_rename(input_dir, length, start_num, pattern, sort_key, sort_order)
    elif mode == 'suffix':
        change_suffix(input_dir, old_suffix, new_suffix)
    
    print("Finished!")


if __name__ == "__main__":
    main()