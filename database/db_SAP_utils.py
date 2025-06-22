import logging
import os
import json
import datetime
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
prod=True

def load_config(config_file=os.path.join("database", "DB_config.json")):
    """
    Loads the configuration file to get database credentials.

    Args:
    config_file (str): Path to the configuration JSON file (default is 'config.json').

    Returns:
    dict: The configuration values.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def get_connection_string(prod, db_label='SAP',driver='ODBC Driver 18 for SQL Server',config_file=os.path.join("database", "DB_config.json")):
    """
    Generates a connection string to connect to the SQL Server database using config from a JSON file.

    Args:
    prod (bool): If False, connects to the QA environment; if True, connects to the production environment.
    db_label (str): A label to differentiate between multiple DBs (e.g., 'AI').
    driver (str): The ODBC driver to use for the connection. Default is 'ODBC Driver 18 for SQL Server'.
    config_file (str): Path to the configuration JSON file (default is 'config.json').

    Returns:
    str: The connection string to connect to the database.
    """
    config = load_config(config_file)

    if not prod:
        print("QA SAP connecting")
        server = config.get(f'DB_{db_label}_SERVER_QA', 'default_server')
        database = config.get(f'DB_{db_label}_DATABASE_QA', 'default_db')
    else:
        print(f"PROD {db_label} connecting")
        server = config.get(f'DB_{db_label}_SERVER_PROD', 'default_server')
        database = config.get(f'DB_{db_label}_DATABASE_PROD', 'default_db')

    username = config.get(f'DB_{db_label}_USER', 'default_user')
    password = config.get(f'DB_{db_label}_PASSWORD', 'default_password')

    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}&TrustServerCertificate=yes'
    return connection_string

def connect_to_db_with_query(connection_string, query):
    """
    Connects to the database using the given connection string and executes the query - SELECT query.
    The results are returned as a Pandas DataFrame.

    Args:
        connection_string (str): The connection string for the specific database.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the query as a DataFrame, or None in case of an error.
    """
    time_start = datetime.datetime.now()
    logger.info(f"Start working on query at {time_start.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create engine and initiate connection
        engine = create_engine(connection_string)
        chunks = []

        # Use tqdm for progress tracking while fetching chunks
        with tqdm(desc="Fetching data", unit="chunk", dynamic_ncols=True) as pbar:
            for i, chunk in enumerate(pd.read_sql(query, engine, chunksize=1000)):
                chunks.append(chunk)
                pbar.update(1)
                pbar.total = i + 1  # Update total dynamically

        # Concatenate all chunks into one DataFrame
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Query completed successfully, {len(df)} rows fetched.")
        return df

    except Exception as e:
        # Logging the error with more context
        logger.error(f"Error while executing query: {e}")
        return None

def convert_encoding(df_query, table_name):
    """
    Convert encoding for specific columns based on the table name in the SQL query.

    Parameters:
    df_query (pd.DataFrame): DataFrame containing query results.
    table_name (str): Table name.

    Returns:
    pd.DataFrame: Updated DataFrame with encoding conversion applied to relevant columns.
    """

    # Mapping of table names to columns that require encoding conversion
    table_columns_map = {
        "DD03M": ['DDTEXT', 'REPTEXT', 'SCRTEXT_S', 'SCRTEXT_M', 'SCRTEXT_L'],
        "KNA1": ['NAME1', 'NAME2', 'ORT01', 'SORTL', 'MCOD1', 'MCOD2', 'MCOD3', 'STRAS'],
        "LFA1": ['NAME1', 'NAME2', 'NAME3', 'NAME4', 'ORT01', 'SORTL', 'STRAS', 'MCOD1', 'MCOD3', 'MCOD2', 'KONZS', 'ANRED'],
        "ZTSD031": ['MAKTX', 'MATNR_MITADEF'],
        "MAKT": ['MAKTG', 'MAKTX'],
        "VBAK": ['KTEXT', 'BSTNK', 'BNAME', 'ZZLIFNR_NAME', 'KDMAT'],
        "VBAP": ['ZZCHAZARA_LMLAY_SACHIR', 'ARKTX', 'KDMAT'],
        "EKPO": ['TXZ01'],
        "T023T": ['WGBEZ'],
        "TSPAT": ['VTEXT'],
        "ZTSD066": ['PROCEDURE_NAME', 'KIT_MATNR_DESC'],
        "ZTMM035": ['UOM', 'MSEHL'],
        "ZTSD044": ['VRKME_IN'],
        "T006A": ['MSEH3', 'MSEH6', 'MSEHT', 'MSEHL'],
        "ZTSD063": ['HOSPITAL_NAME', 'DEPARTMENT_NAME', 'ROOM_ID', 'PROCEDURE_NAME', 'SURGEON_NAME'],
        "ZSD_MTL_FAMREL_V": ['FAM_DESC','TEXT','FAM_DESC_UPPER','FAM_COMMENTS'],
        "MARA": ['MFRPN'],
        "AUSP": ['ATWRT'],
    }

    # Convert table_name to uppercase for case-insensitivity
    table_name_upper = table_name.upper()

    # Get columns that need encoding conversion
    columns_to_convert = set(table_columns_map.get(table_name_upper, [])) #defualt empty list

    # Find intersection of required columns and DataFrame columns
    columns_to_convert = columns_to_convert.intersection(df_query.columns)
    print(f'The columns_to_convert={columns_to_convert}')

    # Apply encoding conversion only on relevant columns
    for column in columns_to_convert:
        df_query[column] = df_query[column].apply(
            lambda x: x.encode('cp850').decode('Windows-1255', errors='replace') if isinstance(x, str) else x
        )

    return df_query

def field_for_CE1SARL(table_name):
    field_string=""
    if table_name=='CE1SARL':
        fields = [
            'PERIO','PALEDGER', 'PAOBJNR', 'GJAHR', 'PERDE', 'BUDAT', 'KNDNR', 'ARTNR', 'FKART', 'KAUFN',
            'FRWAE', 'KURSF', 'KDPOS', 'WERKS', 'VKORG', 'VRGAR', 'VTWEG', 'SPART', 'RBELN',
            'PRCTR', 'MATKL', 'PSTYV', 'HIE01', 'HIE02', 'HIE03', 'PAPH1', 'PAPH2', 'PAPH3',
            'PAPH4', 'PRODH', 'WWLFR', 'ABSMG_ME', 'ABSMG', 'ERLOS', 'VVCGS', 'VVDIS', 'VVFOC',
            'VVSUR', 'VVANX', 'VVSRV', 'VVBON', 'VVBQT_ME', 'VVBQT', 'VVBNV', 'VVH01', 'VVH09'
        ]
        # Join the field names into a single string
        field_string = ", ".join(fields)
    return field_string

def get_query(table_name, prod,perio='',ls_str=""):
    """
    Retrieve an SQL query based on a predefined name and environment.

    Parameters:
    table_name (str): The table name to find the  query to retrieve.
    prod (bool): The environment to use ("prod" for production or QA). Default is "prod".
    ls_str(list): for specif query
    perio=2025001
    Returns:
    str: The SQL query with the correct schema.
    """

    # Define schema mapping based on environment
    schema = "[SRP].[srp]" if prod else "[SRT].[srt]"

    field_string=field_for_CE1SARL('CE1SARL')

    # Dictionary mapping query names to actual queries with placeholders for schema
    queries = {
        'CE1SARL': f"SELECT {field_string}, v.KUNNR FROM (SELECT {field_string} FROM {schema}.CE1SARL WHERE PERIO={perio} AND VRGAR='F') AS c "
                                f"LEFT JOIN {schema}.VBPA AS v ON v.VBELN = c.KAUFN AND v.PARVW = 'WE'",
        'VBFA':f"SELECT VBELN, VBELV ,RFMNG,MEINS, VBTYP_V FROM [SRP].[srp].[VBFA] where VBTYP_V='M' and VBELN IN {ls_str}",
        'A501': f"SELECT [A501].MANDT AS A_MANDT, [A501].KAPPL AS A_KAPPL, [A501].KSCHL AS A_KSCHL, MATNR, DATBI, DATAB, "
                     f"[A501].KNUMH AS A_KNUMH, [KONP].* FROM {schema}.A501 "
                     f"JOIN {schema}.KONP ON [A501].[KNUMH] = [KONP].[KNUMH]",
        'A703': f"SELECT [A703].MANDT AS A_MANDT, [A703].KAPPL AS A_KAPPL, [A703].KSCHL AS A_KSCHL, HIENR, MATNR, "
                     f"[A703].KFRST AS A_KFRST, DATBI, DATAB, [A703].KBSTAT AS A_KBSTAT, [A703].KNUMH AS A_KNUMH, [KONP].* "
                     f"FROM {schema}.A703 "
                     f"JOIN {schema}.KONP ON [A703].[KNUMH] = [KONP].[KNUMH]",
        'A503': f"SELECT [A503].MANDT AS A_MANDT, [A503].KAPPL AS A_KAPPL, [A503].KSCHL AS A_KSCHL, "
                     f"HIENR, MATNR, DATBI, DATAB, [A503].KNUMH AS A_KNUMH, [KONP].* "
                     f"FROM {schema}.A503 JOIN {schema}.KONP ON [A503].[KNUMH] = [KONP].[KNUMH]",
        'MARA': f'SELECT MATNR, SPART, MATKL, MTPOS_MARA, MSTAE, MSTDE, MSTDV, MFRPN, MFRNR, ERSDA, LAEDA, BSTME, MEINS FROM {schema}.MARA',
        'MAKT': f"SELECT MATNR,MAKTX FROM {schema}.MAKT WHERE SPRAS='B'",
        'T023T': f"SELECT MATKL, WGBEZ FROM {schema}.T023T WHERE SPRAS='B'",
        'TSPAT': f"SELECT SPART, VTEXT FROM {schema}.TSPAT WHERE SPRAS='B'",
        'AUSP': f"SELECT OBJEK, ATINN, ATWRT FROM {schema}.AUSP WHERE ATINN IN ('0000000824','0000000825')",
        'KNA1': f'SELECT KUNNR, NAME1, KTOKD, UPDAT, ERDAT FROM {schema}.KNA1',
        'LFA1': f'SELECT ERDAT, LIFNR,LAND1,NAME1 FROM {schema}.LFA1',
        'ZTSD044': f"SELECT VRKME_IN, VRKME_OUT, MSEHL FROM {schema}.ZTSD044",
        'T006A': f"SELECT * FROM {schema}.T006A WHERE SPRAS='B'",
        'ZTMM035': f"SELECT UOM, MSEHL, PER FROM {schema}.ZTMM035",
        "ZSD_MTL_FAMREL_V":f'SELECT * FROM {schema}.[ZSD_MTL_FAMREL_V]',
    }

    # Return the query if found, otherwise raise an error
    return queries.get(table_name, f"Error: Query '{table_name}' not found.")

def export(prod, table_name,perio='', ls_str=''):
    """
    Exports data from a specific table in the SAP database, applies encoding conversion, and returns the DataFrame.
    :param prod:
    :param table_name:
    :return: the new table from the DB after the encoding
    """
    conn_str = get_connection_string(prod,  db_label='SAP')
    query = get_query(table_name, prod, perio, ls_str)
    print(f"query={query}")
    df_query = connect_to_db_with_query(conn_str, query)
    df_query_encoding = convert_encoding(df_query, table_name=table_name)
    return df_query_encoding

if __name__ == "__main__":
    table_name = 'ZSD_MTL_FAMREL_V'
    df_family = export(prod, table_name)