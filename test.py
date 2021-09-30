import numpy
import os
import shutil


"""
def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        print("yes")
    else:
        print("NO")


is_number('3.jpg')
is_number('american_pit_bull_terrier_125.jpg')
is_number('125.jpg')
"""


def main():
    file_list = os.listdir('OxfordIIIT/annotations/xmls')
    abs_path = os.getcwd()
    src_path = abs_path + '\\oxford\\annotations\\xmls'
    for file in file_list:
        src = src_path + str(file)
        p = numpy.random.uniform()
        if p < 0.102:
            dst = abs_path + '\\xml\\test'
        else:
            dst = abs_path + '\\xml\\train'
        shutil.copy(src, dst)


if __name__ == '__main__':
    main()
