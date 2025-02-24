import pandas as pd
import openpyxl


def datload(uploaddocs,filename):
    if filename.endswith('.csv'):
        data= pd.read_csv(uploaddocs)
    elif filename.endswith('.xlsx'):
        data= pd.read_excel(uploaddocs)
    return data