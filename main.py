from defines import *
from excel_to_data_set import DataSetGenerator


def main():


    df = pd.ExcelFile(r'/Users/regevazran/Desktop/technion/semester i/project c/AWHA1 features.xls')
    file = open(df,'r')
    lines = file.readlines()
    for line in lines :
        print(line)

    return
    data_set = None
    excel_path = [r'/Users/regevazran/Desktop/technion/semester i/project c/AWHA1 features.xls']
    data_generator = DataSetGenerator(path=excel_path, dataset=data_set)
    data_generator.get_mice_name_list()
    data_generator.update_mice_features()
    data_generator.add_mice_to_dataset()





if __name__ == '__main__':
    main()
