from defines import *





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_excel(r'/Users/regevazran/Desktop/technion/semester i/project c/AWHA1 features.xls')
    print(df.to_string())
    print(df["Mice"])
