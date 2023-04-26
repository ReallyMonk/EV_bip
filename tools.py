import pandas as pd
import inspect
import re
from functools import wraps
import os

# save files


def write_lines(dest_path, data):
    with open(dest_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + '\n')


def write_json(dest_path, data):
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(data)


def write_numeric(dest_path, data):
    with open(dest_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + ' ')
        f.write('\n')


def reset_file(dest_path):
    with open(dest_path, 'r+') as f:
        f.truncate(0)


class file_wirter():

    def __init__(self, root_path):
        # make sure root path exist
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

    def write_lines(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(str(item) + '\n')

    def write_json(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)

    def write_list(self, filename, data):
        with open(filename, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(str(item) + ' ')
            f.write('\n')

    def reset_file(self, filename):
        # clear all content in file
        with open(filename, 'r+') as f:
            f.truncate(0)


# check parameters
def check_para(para):

    def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]

    para_name = retrieve_name(para)

    print(para_name, type(para), '\n', para)


# print(check_para.__doc__)
