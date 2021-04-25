from defines import *
from excel_to_data_set import DataSetGenerator


def main():

    df = pd.read_excel(r'/Users/regevazran/Desktop/technion/semester i/project c/temp.xlsx',engine ='openpyxl')

    print(df.to_string())




    return
    data_set = None
    excel_path = [r'/Users/regevazran/Desktop/technion/semester i/project c/AWHA1 features.xls']
    data_generator = DataSetGenerator(path=excel_path, dataset=data_set)
    data_generator.get_mice_name_list()
    data_generator.update_mice_features()
    data_generator.add_mice_to_dataset()





if __name__ == '__main__':
    main()
