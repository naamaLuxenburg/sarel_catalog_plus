import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, insert, Table, delete,text
from sqlalchemy.exc import IntegrityError,ProgrammingError
import datetime
from tqdm import tqdm
import logging
import os
import sys
# Add root directory to sys.path safely for both script and interactive environments
try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    print(f"Root path set to: {root_path}")

except NameError:
    # __file__ is not defined in Interactive Window
    root_path = os.path.abspath(os.path.join(os.getcwd(), '../'))

if root_path not in sys.path:
    sys.path.append(root_path)
    print(f"Added root path to sys.path: {root_path}")

from database.db_SAP_utils import load_config, get_connection_string,connect_to_db_with_query
from app.constants import *
print(f"✅ constants.py imported successfully {final_desc}")

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
prod=True

def replace_empty_with_null_safe(df, ls_drop):
    """
    Cleans a DataFrame to ensure compatibility with SQL NULL values by replacing empty strings and NaNs
    with appropriate Python null types.

    Goals:
        ✔ Prepare DataFrame for safe SQL insertion where empty strings and NaNs are treated as SQL NULLs.
        ✔ Ensure consistent null handling across string, numeric, and other column types.

    Behavior:
        - For string columns (excluding those in `ls_drop`): 
          Empty strings ("") and whitespace-only values are replaced with Python `None`.
        - For numeric and other non-string columns (excluding those in `ls_drop`): 
          NaN values are replaced with `pd.NA` (which translates well to SQL NULL).
        - Columns listed in `ls_drop` (default: ['inserted_at']) are excluded from changes.

    Parameters:
        df (pd.DataFrame): The input DataFrame to clean.
        ls_drop (list): Column names to exclude from the replacement logic. Default is ['inserted_at'].

    Returns:
        pd.DataFrame: The cleaned DataFrame, ready for SQL-safe export with NULL-compatible values.
    
    """
    for col in [inserted_date,updated_date,prompt_date]:
        if col not in ls_drop:
            ls_drop.append(col)

    # Select only the columns that are of string type and not in ls_drop
    str_cols = [col for col in df.select_dtypes(include=['object', 'string']).columns if col not in ls_drop]
    df[str_cols] = df[str_cols].applymap(
        lambda x: None if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x)

    # Replace NaNs in non-object columns (like float, int) with None
    num_cols = [col for col in df.columns if col not in ls_drop and col not in str_cols]
    # print(f'the column to change nan values are : {num_cols}')
    # Replace NaN with pd.NA (nullable missing value for numeric columns)
    for col in num_cols:
        df[col] = df[col].apply(lambda x: pd.NA if pd.isna(x) else x)

    return df

def inserted_column(df, flag_insert, flag_update, flag_prompt=False):
    if flag_insert:
        #df['inserted_at'] = datetime.datetime.now().date()
        df[inserted_date] =datetime.datetime.now().replace(microsecond=0)
    if flag_update:
        #df['updated_at'] = datetime.datetime.now().date()
        df[updated_date] = datetime.datetime.now().replace(microsecond=0)
    if flag_prompt:
        #df['updated_at'] = datetime.datetime.now().date()
        df[prompt_date] = datetime.datetime.now().replace(microsecond=0)
    return df

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


def get_table_AI(table_name,db_label, ls_field=[],ls_year=[]):
    """
    The function get the table data from the AIDB.
    The table_name give as th query
    :param conn_str:
    :param table_name:
    :return:
    """
    conn_str = get_connection_string(prod=prod, db_label=db_label)
    query = get_query_AI(table_name,ls_field, ls_year)
    if query is None or str(query).startswith("Error"):
        print(f"[ERROR] Failed to get query: {query}")
        return None  # or raise an exception if preferred
    print(f"query={query}")
    df_query = connect_to_db_with_query(conn_str, query)
    return df_query

def table_exists(table_name, connection_string):
    """Check if a table exists in the target database.
    return: bool, True if the table exists, False otherwise.
    """
    engine = create_engine(connection_string)
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def create_dataframe_to_table(df, table_name, connection_string, if_exists='append'):
    """
    Create andInserts a DataFrame into the specified SQL table.

    Args:
        df (pd.DataFrame): DataFrame to insert.
        table_name (str): Target table name.
        connection_string (str): SQLAlchemy connection string.
        if_exists (str): if the table already exist:
        'fail' - Raises an error if table exists
        'replace' - Drops the table and recreates it before insert
        'append' (default). Inserts rows into the existing table
    """

    try:
        engine = create_engine(connection_string)
        with engine.begin() as conn:
            df.to_sql(table_name, con=conn, if_exists=if_exists, index=False)
        action = {
            'fail': 'attempted insert (fail if exists)',
            'replace': 'replaced the table',
            'append': 'create/appended to the table'
        }.get(if_exists, 'performed an unknown action')

        logging.info(f"✅ Successfully {action}: {len(df)} rows into '{table_name}'.")
    except Exception as e:
        logging.error(f"❌ Failed to insert into {table_name}: {e}")


def insert_dataframe_safe(df, table_name, connection_string, replace_all=False):
    """Safely inserts or replaces data in an existing SQL table without modifying schema."""
    engine = create_engine(connection_string)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables.get(table_name)

    if table is None:
        raise ValueError(f"Table '{table_name}' does not exist.")

    with engine.begin() as conn:
        if replace_all:
            delete_result = conn.execute(delete(table))
            logging.info(f"🗑️ Deleted {delete_result.rowcount} existing rows from '{table_name}' before insert.")

        try:
            # Try to bulk insert first
            conn.execute(insert(table), df.to_dict(orient='records'))
            logging.info(f"✅ Inserted {len(df)} rows into '{table_name}'.")
        except (ProgrammingError, IntegrityError) as e:
            logging.warning("⚠️ Bulk insert failed. Attempting row-by-row insert to isolate errors...")
            success_count = 0
            fail_count = 0
            for i, row in df.iterrows():
                try:
                    conn.execute(insert(table), [row.to_dict()])
                    success_count += 1
                except IntegrityError as e_row:
                    keys = [f"{col}={row[col]}" for col in table.primary_key.columns.keys() if col in row]
                    print(f"{', '.join(keys)} already in the DB (row {i})")
                    fail_count += 1
                    continue  # Skip and continue with next row
                except ProgrammingError as e_row:
                    print(f"\n❌ Row {i} failed to insert due to ProgrammingError.")
                    for col, val in row.items():
                        try:
                            float(val)
                        except (ValueError, TypeError):
                            print(f"   🔎 Column '{col}' has problematic value: {val} ({type(val)})")
                    print(f"   🧨 Exception: {e_row}")
                    raise e_row  # Optional: re-raise if this is critical

            logging.info(f"✅ Finished insert: {success_count} rows inserted, {fail_count} rows skipped.")


def update_dataframe_rows(df, table_name, connection_string, key_columns, update_columns):
    """
    Updates existing rows in a SQL table using values from a DataFrame.

    This version allows:
    - Multiple key columns for identifying the row(s)
    - Multiple columns to update (plus 'updated_at', which is always updated)

    Args:
        df (pd.DataFrame): DataFrame containing the data to update.
        table_name (str): SQL table to update.
        connection_string (str): SQLAlchemy-compatible DB connection string.
        key_columns (list of str): Column(s) used to match existing rows.
        update_columns (list of str): Column(s) to be updated.
    """
    # 🔁 Normalize inputs to lists
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    if isinstance(update_columns, str):
        update_columns = [update_columns]

    engine = create_engine(connection_string)
    updated_count = 0

    # Add 'updated_at' to the update list (enforced)
    if updated_date not in update_columns:
        all_update_columns = update_columns + updated_date
    else:
        all_update_columns=update_columns
    print(f'The update columns are {all_update_columns}')

    with engine.begin() as conn:
        for idx, row in df.iterrows():
            try:
                # Build dynamic SET clause
                set_clause = ', '.join([f"{col} = :{col}" for col in all_update_columns])
                # print(f'set_clause={set_clause}')

                # Build dynamic WHERE clause
                where_clause = ' AND '.join([f"{col} = :{col}" for col in key_columns])
                # print(f'where_clause={where_clause}')

                sql = text(f"""
                    UPDATE {table_name}
                    SET {set_clause}
                    WHERE {where_clause}
                """)
                # print(f'sql={sql}')


                # Prepare the parameter dictionary
                # params = {col: row[col] for col in all_update_columns + key_columns}
                # print(f'params={params}')
                params = {col: (None if pd.isna(row[col]) else row[col]) for col in all_update_columns + key_columns}
                # print(f'params clean={params}')

                result = conn.execute(sql, params)
                # print(f'result={result}')

                if result.rowcount > 0:
                    updated_count += 1

            except Exception as e:
                logging.warning(f"⚠️ Failed to update row {idx} ({', '.join(f'{k}={row[k]}' for k in key_columns)}): {e}")

    logging.info(f"✅ Updated {updated_count} rows in '{table_name}'.")


def load_dataframe_to_table(df, db_label, table_name, mode='append', auto_create_if_missing=True,update_columns=updated_date,key_columns='MATNR'):
    """
    Main controller function to insert DataFrame to SQL table.
    This function intelligently handles the insertion process by:
    - Creating a new table (`mode='create'`)
    - Appending rows to an existing table (`mode='append'`)
    - Replacing all existing rows while preserving the schema (`mode='replace'`)
    Optionally, it can auto-create the table if it does not exist and the mode is 'append' or 'replace'.


    Args:
        df (pd.DataFrame): The data to insert.
        table_name (str): Target table name.
        connection_string (str): SQLAlchemy connection string.
        mode : str, default='append'
        Determines the insert strategy:
        - 'create': Drops the table if it exists, then creates a new one using the DataFrame schema.
        - 'append': Appends data to an existing table without modifying its schema.
        - 'replace': Deletes all existing rows from the table before inserting the new data.
        auto_create_if_missing (bool): If True, will create the table if it doesn't exist.
    """
    connection_string = get_connection_string(prod=prod, db_label=db_label)
    table_exists_flag = table_exists(table_name, connection_string)

    if mode == 'create':
        logging.info(f"🆕 Creating/replacing table '{table_name}'.")
        create_dataframe_to_table(df, table_name, connection_string)

    elif mode in ['append', 'replace']:
        if not table_exists_flag:
            if auto_create_if_missing:
                logging.warning(f"⚠️ Table '{table_name}' does not exist. Falling back to mode='create'.")
                create_dataframe_to_table(df, table_name, connection_string)
                return
            else:
                raise ValueError(
                    f"❌ Table '{table_name}' does not exist. Set auto_create_if_missing=True to create it.")

        if mode == 'append':
            logging.info(f"➕ Appending data to '{table_name}'.")
            insert_dataframe_safe(df, table_name, connection_string, replace_all=False)
      
        elif mode == 'replace':
            logging.info(f"🔁 Replacing data in '{table_name}'.")
            try:
                insert_dataframe_safe(df, table_name, connection_string, replace_all=True)
            except IntegrityError as e:
                logging.error(f"❌ IntegrityError occurred while replacing data in '{table_name}': {e.orig}")
                # Handle the integrity issue for replacing data
                print(f"⚠️ Integrity issue detected while replacing data.")

    elif mode == 'update':
        logging.info(f"🛠️ Updating existing rows in '{table_name}' based on {key_columns}.")
        update_dataframe_rows(df, table_name, connection_string, key_columns, update_columns)

    else:
        raise ValueError("❌ Invalid mode. Use 'create', 'append', or 'replace'.")




if __name__ == "__main__":
    df_org=get_table_AI('CE1SARL_Invoice', [], [2025])
