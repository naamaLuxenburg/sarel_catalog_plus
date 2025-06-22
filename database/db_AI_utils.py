import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, insert, Table, delete,text
from sqlalchemy.exc import IntegrityError
import datetime
from tqdm import tqdm
import logging
import os
import sys

# Add the root directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_SAP_utils import load_config, get_connection_string,connect_to_db_with_query

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
prod=True


def get_query_AI(table_name, ls_field,ls_year):
    if len(ls_field)==0:
        if table_name.startswith('CE1SARL_Invoice'):
            str_field = (
                'BUDAT, PERIO, GJAHR,PERDE, KNDNR, ARTNR,RBELN,KUNNR,SALES,ERLOS,'
                'ABSMG_ME, ABSMG,Code_number_ABSMG, MSEHL_ABSMG,amount_ABSMG,unit_price_ABSMG,'
                'VVBQT_ME,VVBQT,Code_number_VVBQT,MSEHL_VVBQT,amount_VVBQT, unit_price_VVBQT,'
                'KAUFN, WERKS, SPART, MATKL, DATBI,DATAB,KBETR, KONWA, KMEIN, Org_table,HIENR,inserted_at')
        else:
            str_field = '*'
    else:
        str_field = ', '.join(ls_field)
    str_years = ', '.join(str(y) for y in ls_year)
    yesterday_date = datetime.date.today() - datetime.timedelta(days=1)
    str_date=str(yesterday_date)


    queries = {
        'MARA_Products': f'SELECT {str_field} FROM [AIDB].[dbo].[MARA_Products]',
        'CE1SARL_Invoice':f'SELECT {str_field} FROM [AIDB].[dbo].[CE1SARL_Invoice] where GJAHR IN ({str_years})',
        'T023T_MATKL_description': f"SELECT {str_field} FROM [AIDB].[dbo].[T023T_MATKL_description]",
        'TSPAT_dv_description': f"SELECT {str_field} FROM [AIDB].[dbo].[TSPAT_dv_description]",
        'KNA1_Customer': f'SELECT {str_field} FROM [AIDB].[dbo].[KNA1_Customer]',
        'LFA1_Suppliers': f'SELECT {str_field} FROM [AIDB].[dbo].[LFA1_Suppliers]',
        'T006A_ZTMM035_ZTSD044_Code_units': f'SELECT {str_field} FROM [AIDB].[dbo].[T006A_ZTMM035_ZTSD044_Code_units]',
        'A501_A703_A503_Prices': f'SELECT {str_field} FROM [AIDB].[dbo].[A501_A703_A503_Prices]',
        'TCURR_Coins':f"SELECT {str_field} FROM [AIDB].[dbo].[TCURR_Coins] where CAST(value_valid_from AS DATE) ='{str_date}'",
        'Medical_equipment_product_data': f'SELECT {str_field} FROM [AIDB].[dbo].[Medical_equipment_product_data]',
        'Medical_equipment_groups': f'SELECT {str_field} FROM [AIDB].[dbo].[Medical_equipment_groups]',
        'ZSD_MTL_FAMREL_V_Family': f'SELECT {str_field} FROM [AIDB].[dbo].[ZSD_MTL_FAMREL_V_Family]',
        'Med_data':f'SELECT {str_field} FROM [AIDB].[dbo].[Med_data]',
        'Med_groups':f'SELECT {str_field} FROM [AIDB].[dbo].[Med_groups]',
        'Med_generic_alter':f'SELECT {str_field} FROM [AIDB].[dbo].[Med_generic_alter]'
    }
    # Return the query if found, otherwise raise an error
    #return queries.get(table_name, f"Error: Query '{table_name}' not found.")
    # Return specific query if found
    if table_name in queries:
        return queries[table_name]

    # 🆗 Default fallback if table not predefined
    default_query = f"SELECT {str_field} FROM [AIDB].[dbo].[{table_name}]"
    return default_query


def get_table_AI(table_name, ls_field=[],ls_year=[]):
    """
    The function get the table data from the AIDB.
    The table_name give as th query
    :param conn_str:
    :param table_name:
    :return:
    """
    conn_str = get_connection_string(prod=prod, db_label='AI')
    query = get_query_AI(table_name,ls_field, ls_year)
    if query is None or str(query).startswith("Error"):
        print(f"[ERROR] Failed to get query: {query}")
        return None  # or raise an exception if preferred
    print(f"query={query}")
    df_query = connect_to_db_with_query(conn_str, query)
    return df_query

if __name__ == "__main__":
    df_org=get_table_AI('CE1SARL_Invoice', [], [2025])
