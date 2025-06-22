import datetime
import os
import sys
import re
import numpy as np
import pandas as pd
# from symspellpy.symspellpy import SymSpell, Verbosity

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_AI_utils import *

#enviroment variables
product_id_CE1SARL='ARTNR'

product_id_MARA='MATNR'
product_desc='full_desc'
product_desc_update='full_desc_update'
product_code_sub_field='MATKL'
product_desc_sub_field='WGBEZ'
product_dv_id='SPART'
product_dv_desc='VTEXT'
product_category='MTPOS_MARA'
product_basic_unit='MEINS'
product_order_unit='BSTME'
product_order_basic_unit='BSTME_merge_MEINS'
manufacturer_model='MFRPN'
supplier_name='NAME1'

unit_code='Code1'
unit_desc='MSEHL_x'
unit_code_number='Code_number'
ls_dvs=['11','12','22','25']

def filter_products(df_MARA, df_CE1SARL):

    """
    Filters the DataFrame to include only products:
    - from specified divisions.
    - from specific years
    - where the product ID starts with a digit.

    :param df: DataFrame containing product data.
    :return: Filtered DataFrame containing only products from the specified divisions.
    """
    ls_ARTNR_update = df_CE1SARL[product_id_CE1SARL].unique().tolist()
    ls_MATNR = df_MARA[product_id_MARA].unique().tolist()
    ls_missing = set(ls_ARTNR_update) - set(ls_MATNR)
    if len(ls_missing) != 0:  # if there is mismatch somthing isnt update correctly.
        print('Missing product from the MARA tables, somthing wrong')

    df_updated_products = df_MARA[df_MARA[product_id_MARA].isin(ls_ARTNR_update) &
                                  df_MARA[product_dv_id].isin(ls_dvs)].reset_index(drop=True)
    df_updated_products = df_updated_products[~df_updated_products[product_desc].apply(contains_hebrew)]
    print(f"{len(df_updated_products)} is the number of products that bought in the last 3 years, in the selected divisions and only with English descriptions.")
    return df_updated_products

def merge_MARA_dv_desc(df_MARA,df_T023T, df_TSPAT):
    """

    :param df_MARA: products table
    :param df_T023T: MATKL description
    :param df_TSPAT:dv description
    :return:merge df with all the description
    """
    df_MARA_merge_T023T = df_MARA.merge(df_T023T, on=product_code_sub_field, how='left')
    df_MRAR_merge_final = df_MARA_merge_T023T.merge(df_TSPAT, on=product_dv_id, how='left')
    return df_MRAR_merge_final

def remove_special_chars_from_end(value):
    if isinstance(value, str):  # Check if the input is a string
        value = re.sub(r'[\s\.,-]+$', '', value)  # Remove special characters from the end
    return value

def remove_manufacturer(desc, manu):
    """
     #Function to remove manufacturer model column from description
    :param desc: description
    :param manu: manufacturer
    :return: clean description or the original one
    """
    # Return as-is if either value is missing or not string-like
    if not isinstance(desc, str):
        return desc
    if not isinstance(manu, str) or manu.strip() == '':
        return desc

    manu = remove_special_chars_from_end(manu)
    desc=remove_special_chars_from_end(desc)
    if manu and desc.endswith(manu):  # Only process if manufacturer is not empty and matches the end of description
        return desc[:-(len(manu))].rstrip()  # Remove manufacturer from end and strip any trailing spaces
    elif manu in desc:
        return desc.replace(manu,"").strip()
    # Check if any part of manu exists in desc (partial match)
    elif manu and any(part in desc for part in re.split(r'[-=.,:/\\\s]', manu)):
        for part in re.split(r'[-=.,:/\\\s]', manu):  # Split manu into meaningful parts

            if part in desc and part:  # Check if part exists and is non-empty - #remove only one. if the same part appear twice it will be again
                #print(f'{part}')
                desc = " ".join(desc.rsplit(part, 1)).strip()  # Replace only the last occurrence

        return desc  # Return modified desc with parts of manu removed

    #still nothing return yet
    desc_manu = desc.split()[-1]
    desc_manu_fix = "".join(desc_manu.split('-'))
    if desc_manu_fix in manu:
        return desc.replace(desc_manu, '').strip()
    elif (desc_manu_fix in manu[:-1] or desc_manu_fix in manu[1:-1]
          or desc_manu in manu[:-1] or desc_manu in manu[1:-1]
            or desc_manu[:-1] in manu or desc_manu[1:-1] in manu ):
        return desc.replace(desc_manu, '').strip()
    #return original desc
    return desc

def remove_unit(desc, num):
    # Return as-is if either value is missing or not string-like
    if not isinstance(desc, str):
        return desc
    code = f'B/{str(num)}'
    # Check for an exact match using word boundaries
    if re.search(rf'\b{re.escape(code)}\b', desc):
        return desc.replace(code, '').strip()
    return desc

def clean_desc(desc, manu, num):
    # Return as-is if either value is missing or not string-like
    if not isinstance(desc, str):
        return desc
    
    #step1 - remove manu
    desc_fix1 = remove_manufacturer(desc, manu)

    #step2 - remove unit
    if pd.notna(num):
        desc_fix2=remove_unit(desc_fix1, int(num))
    else:
        desc_fix2=desc_fix1
        print(f"Warning: num is NaN for description code number , need to update the tables: ZTMM035, ZTSD044,T006A and run fix Konp")

    # step4 - only one space between words
    desc_fix3 = re.sub(r'\s+', ' ', desc_fix2).strip()
    return desc_fix3.strip(' -.')

def fill_missing(df,field, fill_field):
    """

    :param df:
    :param field: the field we need to fill values
    :param fill_field:  the field the we used to fill values.
    :return: fix df.
    """
    df[field] = df[field].mask(df[field] == '', df[fill_field])
    df[field] = df[field].fillna(df[fill_field])
    return df

def create_update_desc_without_manu_unit(df, original_col, update_suffix='_update'):
    """
    Creates an updated version of a column using `clean_desc`, inserts it after the original column,
    and fills any missing values in the new column using the original column as a fallback.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - original_col (str): The name of the original column to update.
    - update_suffix (str): Suffix for the new column name (default is '_update').

    Returns:
    - pd.DataFrame: The updated DataFrame with the new column added and missing values filled.
    """
    update_col = original_col + update_suffix

    # Apply the clean_desc transformation
    df[update_col] = df.apply(
        lambda row: clean_desc(row[original_col], row[manufacturer_model], row[unit_code_number]),
        axis=1
    )

    # Insert the new column right after the original column
    col_index = df.columns.get_loc(original_col) + 1
    cols = list(df.columns)
    cols.insert(col_index, cols.pop(cols.index(update_col)))
    df = df[cols]

    # Fill missing values in the new column using the original as fallback
    df = fill_missing(df, update_col, original_col)
    return df

def add_MARA_products_fields(df_MARA, df_T023T, df_TSPAT, df_codes):
    """

    :param df_MARA: org from SQL
    :param df_T023T: org from SQL
    :param df_TSPAT: org from SQL
    :param df_codes: org from SQL
    :return: new df with merge column, new column description, filter according the products that are relevant from the past years.
    """
    ls_filter_fields = [product_id_MARA, product_desc,product_dv_id,product_dv_desc,
                        product_code_sub_field,product_desc_sub_field, product_category,
                        product_basic_unit, product_order_unit,
                        supplier_name, manufacturer_model]
    #add to MARA the description of the dv and the MATKL
    df_MARA_desc = merge_MARA_dv_desc(df_MARA, df_T023T, df_TSPAT)
    # Filter the DataFrame to include only the relevant columns
    df_products = df_MARA_desc[ls_filter_fields]
    # Replace None or empty strings in 'BSTME' with the value from 'MEINS'
    df_products[product_order_basic_unit] = df_products[product_order_unit].where(
        df_products[product_order_unit].notnull() & (df_products[product_order_unit] != ' '),
        df_products[product_basic_unit]
    )
    df_products = df_products.merge(df_codes[[unit_code, unit_desc, unit_code_number]], left_on=product_order_basic_unit,
                                    right_on=unit_code, how='left')
    df_products = df_products.drop(columns=[unit_code, product_basic_unit, product_order_unit])
    # df_products['MATKL_4'] = df_products['MATKL'].apply(get_PAPH3)

    #filter out rows where the MATNR starts with a digit
    # df_field = df_field[df_field[product_id_MARA].str.match(r'^\d')].reset_index(drop=True)
    
    #add the new update description columns
    df_products_final = create_update_desc_without_manu_unit(df_products, product_desc)
    return df_products_final

def contains_hebrew(text):
    # Function to check if a string contains any Hebrew letters
    return bool(re.search(r'[\u0590-\u05FF]', str(text)))


#region Step init
current_year=datetime.datetime.now().year
years=np.arange(2023,current_year+1)

print("Reading tables from DBAI...")
df_CE1SARL = get_table_AI('CE1SARL_Invoice', [], years)
df_MARA =get_table_AI('MARA_Products')
df_T023T=get_table_AI('T023T_MATKL_description')
df_TSPAT=get_table_AI('TSPAT_dv_description')
df_codes =get_table_AI('T006A_ZTMM035_ZTSD044_Code_units')
print("Tables read successfully.")




df_MARA_fields=add_MARA_products_fields(df_MARA, df_T023T, df_TSPAT,df_codes)
df_updated_products = filter_products(df_MARA_fields, df_CE1SARL)


random_sample = df_updated_products[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]].sample(n=100, random_state=42)
random_sample.to_excel('temp_excel/random_sample.xlsx', index=False)

#endregion


#region step a

import scispacy
import spacy

# Load the scispaCy model
nlp = spacy.load("en_core_sci_sm")





#endregion