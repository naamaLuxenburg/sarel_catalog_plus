import pandas as pd
from collections import OrderedDict
import os
import sys
from rapidfuzz import process, fuzz
import importlib
import re



# # Add root directory to sys.path safely for both script and interactive environments
# try:
#     root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
#     print(f"Root path set to: {root_path}")
# except NameError:
#     # __file__ is not defined in Interactive Window
#     root_path = os.path.abspath(os.path.join(os.getcwd(), '../../'))

# if root_path not in sys.path:
#     sys.path.append(root_path)
#     print(f"Added root path to sys.path: {root_path}")

# Now safely import
from app.models.process_descriptions import contains_hebrew,get_faiss_neighbors
from app.constants import *
# print(f"✅ constants.py imported successfully {final_desc}")
from database.db_AI_utils import *
# import database.db_AI_utils as db_utils
# importlib.reload(db_utils)


def read_prodcuts_data(flag_db=False):
    """
    Reads all the prodcuts exist data returns a DataFrame.
    """
    if flag_db:
        print(f"Reading data from the database")
        df_MARA =get_table_AI('MARA_Products', 'AI')
        df_med =get_table_AI('Med_data', 'AI')
        df_update_price =get_table_AI(table_name='A501_A703_A503_Updated_prices',db_label='AI')
    else:
        print(f"Reading data from parquet files in {workspace_root}")
        df_MARA = pd.read_parquet(os.path.join(workspace_root, 'Data/MARA_Products.parquet'))
        df_med = pd.read_parquet(os.path.join(workspace_root, 'Data/Med_data.parquet'))
        df_update_price = pd.read_parquet(os.path.join(workspace_root, 'Data/A501_A703_A503_Updated_prices.parquet'))
    df_update_price_filter=df_update_price[df_update_price[price_client]=='X'].reset_index(drop=True) #only General Sarel price list
    cols_med= [product_id_MARA,med_generic,med_concentration,med_dosage, med_dose]
    cols_price=[product_id_MARA,price_in_coin, price_coin, price_unit, price_unit_ILS]
    

    df_products_med=df_MARA.merge(df_med[cols_med], on=product_id_MARA,how='left')
    df_products_price=df_products_med.merge(df_update_price_filter[cols_price], on=product_id_MARA, how='left')


    desc_path = os.path.join(workspace_root, 'app/models/temp_result')
    df_new_desc = pd.read_excel(os.path.join(desc_path, 'df_new_desc.xlsx')) ##it will be from the DB in the future
    df_products_total = df_products_price.merge(df_new_desc, on=product_id_MARA, how='left')

    ##remove later:
    df_products_total[final_desc]=df_products_total[final_desc].fillna(df_products_total[product_desc])

    cols_MARA=[product_id_MARA,product_desc,final_desc,product_dv_id,manufacturer_model,supplier_name, product_basic_unit,product_order_unit]
    cols_all = cols_MARA + cols_med + cols_price
    cols_filtered=list(OrderedDict.fromkeys(cols_all))
    df_products_final=df_products_total[cols_filtered]
    print(f"✅ read_prodcuts_data() completed")
    print(f"Total products in df_products: {len(df_products_final)}")

    return df_products_final

def filtered_input_data(df_input):
    """
    Filters the input DataFrame to remove rows with Hebrew characters in the input_desc column.
    Args:
        df_input (pd.DataFrame): Input DataFrame containing the data to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows containing Hebrew characters removed.
    """
    df_input_heb_filtered = df_input[~df_input[input_desc].apply(contains_hebrew)].reset_index(drop=True)
    df_input_filtered=df_input_heb_filtered.drop_duplicates().reset_index(drop=True)
    df_input_filtered.insert(0, 'input_id', range(1, len(df_input_filtered) + 1))
    df_input_filtered[input_desc_norm]=df_input_filtered[input_desc].str.lower()
    num_input_records = df_input_filtered['input_id'].nunique()
    print(f'Total input records after filtering Hebrew and remove duplicates records: {num_input_records}')
    # df_input_filtered_org= df_input_filtered.copy()

    return df_input_filtered


def get_fuzzy_matches(input_val, reference_list, threshold=85, top_k=5):
    """
    Perform fuzzy matching on the input value against a reference list.
    Args:
        input_val (str): The input value to match. (e.g., manufacturer model number or description)
        reference_list (list): The list of reference values to match against. (e.g., manufacturer model number or description)
        threshold (int): The score threshold for matches.
        top_k (int): The number of top matches to return.

    Returns:
        - top_matches: List of top K matches with their scores 
        - above_threshold_matches: List of matches above the threshold with their scores
        - best_match: The best match from the reference list
        - best_score: The score of the best match
    Notes:
    -----
    If the input value is missing or empty, all outputs are None.
    Uses RapidFuzz with token_sort_ratio for scoring.
    
    """

    # print(f'The th is {threshold}')
    # Check if input_val is empty or NaN 
    if not isinstance(input_val, str) or input_val.strip() == "":
        return None, None,None, None  # Skip if empty or NaN

    
    #score_methods=fuzz.token_set_ratio #also for reordered and Flexible with extra/missing words
    score_methods=fuzz.token_sort_ratio #Good for reordered tokens
    # limit=round(len(reference_list)/3)
    limit=250
    
    # the number of matches depended on the limit.
    all_matches = process.extract(
        input_val,
        reference_list,
        scorer=score_methods, limit=limit
    )
        # Format as list of dicts
    all_matches_dict = sorted(
        [{"match": m[0], "score": m[1]} for m in all_matches],
        key=lambda x: x["score"],
        reverse=True
    )
    # Top K matches (even if score is low)
    top_matches = all_matches_dict[:top_k]
    
    # All matches above threshold
    above_threshold_matches = [m for m in all_matches_dict if m["score"] >= threshold]

        # Best match and score
    best_match = all_matches_dict[0]["match"] if all_matches_dict else None
    best_score = all_matches_dict[0]["score"] if all_matches_dict else None

    return top_matches, above_threshold_matches,best_match, best_score


def apply_fuzzy_match(df, col_input, reference_col, reference_df, suffix="", threshold=85, top_k=5):
    """
    Apply fuzzy matching to a column in a DataFrame and add result columns.
    Args:
        df (pd.DataFrame): DataFrame containing the input column to match.
        col_input (str): Column name in df to perform fuzzy matching on.
        reference_col (str): Column name in reference_df to match against.
        reference_df (pd.DataFrame): DataFrame containing the reference values (to create the reference list).
        suffix (str): Suffix to append to the result columns.
        threshold (int): Score threshold for matches.
        top_k (int): Number of top matches to return.

    Returns:
        Updated DataFrame with:
            - top matches list
            - matches above threshold
            - best match
            - best score
    """
    reference_list = reference_df[reference_col].dropna().unique().tolist()
    # print(f'the ref list is: {reference_list}')
    print(f"🔍 Starting fuzzy match: {col_input} → {reference_col} (threshold={threshold}, top_k={top_k})")
    
    result_cols = [f"{suffix}top_matches", f"{suffix}matches_above_threshold", f"{suffix}best_match", f"{suffix}best_score"]

    df[result_cols] = df[col_input].apply(
        lambda x: pd.Series(get_fuzzy_matches(x, reference_list, threshold=threshold, top_k=top_k))
    )
    
    print(f"✅ Fuzzy match complete. Added columns: {', '.join(result_cols)}\n")
    return df

def create_fuzzy_match_columns_statistics(df_input_filtered, df_products, suffix, threshold=85):
    """
    Create fuzzy match columns and print statistics.
    Args:
        """
    
    flag_apply=True
    if 'manu' in suffix : #manufacturer
        fuzzy_value='provider numbers'
        col_input=input_provider_num
        reference_col= manufacturer_model

    elif 'desc' in suffix: #description
        fuzzy_value='descriptions'
        col_input=input_desc_norm
        reference_col= 'desc_fix'
    else: 
        fuzzy_value='generic'
        col_input=input_desc_norm
        reference_col= 'fix_Generic_Name' #med_generic
    
    if col_input not in df_input_filtered.columns:
        print(f"Column '{col_input}' not found in input data — matching skipped")
        flag_apply=False
  

    if flag_apply:
        num_input_records = df_input_filtered['input_id'].nunique()
        df_input_filtered_update = apply_fuzzy_match(
        df_input_filtered, 
        col_input, 
        reference_col, 
        df_products, 
        suffix=suffix, 
        threshold=threshold
        )

        print(f"📊 First Summary after fuzzy matching according on {fuzzy_value} with threshold={threshold}")
        num_empty = df_input_filtered_update[col_input].isna().sum()
        num_recoreds = len(df_input_filtered_update[col_input].dropna().tolist())
        num_unique_recoreds = len(df_input_filtered_update[col_input].dropna().unique().tolist())
        print(f"Total unique {fuzzy_value}: {num_unique_recoreds} from {num_input_records} records, precentage: {(num_unique_recoreds/num_input_records)*100:.2f}%")
        print(f"Total {fuzzy_value}: {num_recoreds} from {num_input_records} records, precentage: {(num_recoreds/num_input_records)*100:.2f}%")
        print(f"Total empty {fuzzy_value}: {num_empty} from {num_input_records} records, precentage: {(num_empty/num_input_records)*100:.2f}%")
        print(f"best_score {fuzzy_value}: {df_input_filtered[f'{suffix}best_score'].describe()}")
    else:
        df_input_filtered_update=df_input_filtered.copy()

    return df_input_filtered_update



def print_statistics_match(df_final):
    """
    Prints statistics about the final DataFrame after matching.
    """
    print("📊 Match Breakdown Summary")

    total_rows = len(df_final)
    unique_inputs = df_final['input_id'].nunique()
    print(f"• Total rows in final DataFrame: {total_rows}")
    print(f"• Unique input IDs in final DataFrame: {unique_inputs}")

    # 1. Manufacturer info
    df_with_manu = df_final[df_final[input_provider_num].notna()]
    input_id_unique_manu = df_with_manu['input_id'].nunique()
    unique_manu = df_with_manu[input_provider_num].nunique()

    print(f"• {len(df_with_manu)} rows have a manufacturer number ({len(df_with_manu) / total_rows:.2%})")
    print(f"• {input_id_unique_manu} of them is the original manufacturer data ({input_id_unique_manu / unique_inputs:.2%})")
    print(f"• {unique_manu} unique manufacturer input data ({unique_manu / total_rows:.2%})")


    # # 2. Descriptions present
    df_with_desc = df_final[df_final[input_desc].notna()]
    input_id_unique_desc = df_with_desc['input_id'].nunique()
    unique_desc = df_with_desc[input_desc].nunique()

    print(f"• {len(df_with_desc)} rows have valid descriptions (no missing records)")
    print(f"• {input_id_unique_desc} rows have valid descriptions from org data (no missing records)")
    print(f"• {unique_desc} rows with unique descriptions from org data (no missing records)")


    # 3. Full match via manufacturer (Rule 1 - exact match score 100)
    rule1_full_manu = df_final[df_final['rule_case'] == 'rule1_manu_100']
    rule1_full_manu_unique = rule1_full_manu['input_id'].nunique()
    print(f"• {len(rule1_full_manu)} rows have full manufacturer match (score 100) ({len(rule1_full_manu) / unique_manu:.2%} of unique manu matches)")
    print(f"• {rule1_full_manu_unique} unique rows have full manufacturer match (score 100) ({rule1_full_manu_unique / unique_manu:.2%} of unique manu matches)")


    # 5. Full match by description only
    rule1_full_desc = df_final[df_final['rule_case'] == 'rule1_desc_100']
    rule1_full_desc_unique = rule1_full_desc['input_id'].nunique()
    print(f"• {len(rule1_full_desc)} rows have full description match only (score 100)")
    print(f"• {rule1_full_desc_unique} unique rows have full manufacturer match (score 100)")

      # Full match by description and manu
    rule1_full_both = df_final[df_final['rule_case'] == 'rule1_both_100']
    rule1_full_both_unique = rule1_full_both['input_id'].nunique()
    print(f"• {len(rule1_full_both)} rows double full match description and manufacturer (score 100)")
    print(f"• {rule1_full_both_unique} unique rows with double full match description and manufacturer (score 100)")



    # 6. Both matched same product (desc & manu match same product)
    rule2_same_product = df_final[df_final['rule_case'] == 'rule2_same_match']
    print(f"• {len(rule2_same_product)} rows where both manu and desc matched the same product ({len(rule2_same_product) / total_rows:.2%})")


    # 7. Only description score > 75 or no manu
    rule3_desc_only = df_final[df_final['rule_case'] == 'rule3_only_desc']
    print(f"• {len(rule3_desc_only)} rows where only description is above {th_desc} or no manufacturer ({len(rule3_desc_only) / total_rows:.2%})")
    print(f"• {rule3_desc_only[input_desc].nunique()} distinc rows where only description is above {th_desc} or no manufacturer ({len(rule3_desc_only) / total_rows:.2%})")



    # 8. Only manufacturer score > 75
    rule3_manu_only = df_final[df_final['rule_case'] == 'rule3_only_manu']
    print(f"• {len(rule3_manu_only)} rows where only manufacturer score is above {th_manu} ({len(rule3_manu_only) / total_rows:.2%})")
    print(f"• {rule3_manu_only['input_id'].nunique()} original rows where only manufacturer is above {th_manu} or no description ({len(rule3_desc_only) / total_rows:.2%})")
    print(f"• {rule3_manu_only[input_provider_num].nunique()} Distinc rows where only manufacturer is above {th_manu} or no description ({len(rule3_desc_only) / total_rows:.2%})")



    # 9. Both scores above 75 - took higher one
    rule4_both_above_75 = df_final[df_final['rule_case'].isin(['rule4_both_high_manu','rule4_both_high_desc'])]
    print(f"• {len(rule4_both_above_75)} rows where both scores > 75 – picked the higher one")



    # 10. Remaining - below 75 or empty
    rule5_no_match = df_final[df_final['rule_case'].isna()]
    print(f"• {len(rule5_no_match)} rows with no match (both scores < 75 or missing) ({len(rule5_no_match) / total_rows:.2%})")
    print(f"• {rule5_no_match['input_id'].nunique()} org rows with no match (both scores < 75 or missing) ({len(rule5_no_match) / total_rows:.2%})")



def enhanced_breakdown_and_final_match(df, df_products):
    """
    Applies a rule-based breakdown to determine the final product match from fuzzy matching results.
    
    Rules:
    1. Full match (score == 100) for either or both → pick highest, and indicate which.
    2. Same result from both fuzzy searches (manufacturer & description) and 
    their texts match → use that match.
    3. Only one score ≥ 75 → take that one (desc or manu).
    4. Both ≥ 75 → pick the higher score.
    5. Otherwise → no match.

    Args:
        df (pd.DataFrame): DataFrame with fuzzy match results (must include description, best_match, best_score, etc.)
        df_products (pd.DataFrame): Product list with reference descriptions (`desc_fix`) and `manufacturer_model`.

    Returns:
        pd.DataFrame: Input dataframe + 3 new columns:
            - final_match_product (str): The final product match based on rules
            - final_match_desc (str): the final description based on rules
            - final_match_score (int): matching score
            - final_match_source (str): match source - according to 'desc', 'manu', 'both', or 'none'
            -'rule_case' (str): rule case applied for the match accoridng to the rules above. The opinal names:
            1.'rule1_manu_100', 'rule1_desc_100', 'rule1_both_100'
            2. 'rule2_same_match'
            3. 'rule3_only_manu', 'rule3_only_desc'
            4. 'rule4_both_high_manu', 'rule4_both_high_desc'
            5. None - 'rule5_no_match'for no match
        
    """
    # Join to bring the manufacturer description from product reference
    if 'manu_best_match' in df:
        df_products_without_manu_model = df_products[~df_products[manufacturer_model].isna()].reset_index(drop=True)
        df = df.merge(
            df_products_without_manu_model[[manufacturer_model, final_desc, product_id_MARA, product_dv_id]].rename(columns={final_desc: 'manu_desc_match',product_id_MARA: 'manu_product_match',product_dv_id:'manu_dv_match' }),
            how='left',
            left_on='manu_best_match',
            right_on=manufacturer_model
        )
        df.drop(columns=[manufacturer_model], inplace=True)
    else:
        print(f'Manufacturer not in the input file - merge skipped')

    if 'desc_best_match' in df:
        df = df.merge(
            df_products[[manufacturer_model, final_desc, product_id_MARA, product_dv_id]].rename(columns={product_id_MARA: 'desc_product_match', product_dv_id:'desc_dv_match'}),
            how='left',
            left_on='desc_best_match',
            right_on=final_desc
        )
        df.drop(columns=[final_desc], inplace=True)
    else:
        print(f'Description not in the input file - merge skipped')


    def apply_match_logic(row, th_match=85):
        # desc_val = row.get('description', '')
        manu_desc = row.get('manu_desc_match')
        manu_score = row.get('manu_best_score', 0)
        manu_match = row.get('manu_best_match')
        manu_product = row.get('manu_product_match')
        manu_dv = row.get('manu_dv_match')
        # print(f'manu values: manu_desc={manu_desc}, manu_score={manu_score}, manu_match={manu_match}, manu_product={manu_product}, manu_dv={manu_dv}')

        desc_manu=row.get(manufacturer_model)
        desc_score = row.get('desc_best_score', 0)
        desc_match = row.get('desc_best_match')
        desc_product = row.get('desc_product_match')
        desc_dv = row.get('desc_dv_match')
        # print(f'desc values: desc_manu={desc_manu}, desc_score={desc_score}, desc_match={desc_match}, desc_product={desc_product}, desc_dv={desc_dv}')

        manu_source="manu"
        desc_source="desc"
        both_source="both"

        # Rule 1: Score == 100 → source both if equal, else stronger
        if manu_score == 100 and desc_score == 100: #for now assupe both are equal product
            return pd.Series([desc_product, desc_dv,desc_match, desc_manu,100,both_source, 'rule1_both_100'])
        elif manu_score == 100:
            return pd.Series([manu_product, manu_dv,manu_desc,manu_match, 100, manu_source,'rule1_manu_100'])
        elif desc_score == 100:
            return pd.Series([desc_product,desc_dv,desc_match,desc_manu, 100, desc_source,'rule1_desc_100'])

        # Rule 2: same result from both -> descriptions match
        if pd.notna(manu_match) and pd.notna(desc_match) and manu_desc == desc_match:
            both_score= ((manu_score+desc_score)/2)+5 #improve later
            return pd.Series([desc_product,desc_dv, desc_match,desc_manu,  both_score, both_source,'rule2_same_match'])

        # Rule 3: one over 75, other is low or missing
        if manu_score >= th_match and (desc_score < th_match or pd.isna(desc_score)):
            return pd.Series([manu_product, manu_dv, manu_desc,manu_match, manu_score, manu_source,'rule3_only_manu'])
        if desc_score >= th_match and (manu_score < th_match or pd.isna(manu_score)):
            return pd.Series([desc_product, desc_dv, desc_match, desc_manu, desc_score, desc_source,'rule3_only_desc'])

        # Rule 4: both ≥ 75 → choose higher
        if manu_score >= th_match and desc_score >= th_match:
            if manu_score >= desc_score:
                return pd.Series([manu_product, manu_dv, manu_desc,manu_match, manu_score, manu_source, 'rule4_both_high_manu'])
            else:
                return pd.Series([desc_product, desc_dv, desc_match, desc_manu, desc_score, desc_source,'rule4_both_high_desc'])

        # Rule 5: no match
        return pd.Series([None, None, None, None, None, None, None])

    cols_final_match=['final_match_product','final_match_dv','final_match_desc' ,'final_match_manufacturer_model','final_match_score', 'final_match_source','rule_case']
    df[cols_final_match] = df.apply(apply_match_logic, axis=1)        
    
    #print_statistics_match(df)

    # cols_total=['input_id', input_desc,input_supplier_name ,input_provider_num, input_id_number]+cols_final_match

    # df_final=df[cols_total]
    df_final=df.copy()
    
    return df_final

def create_match_product(df_input):
    #step 1 -  read the products data
    df_products=read_prodcuts_data()

    #step 2 - read the input data and filter it
    df_input_filtered=filtered_input_data(df_input)

    #step 3 - apply fuzzy matching and updated df
    #parames
    suffix_manu = "manu_"
    suffix_desc = "desc_"


    df_input_filtered=create_fuzzy_match_columns_statistics(df_input_filtered, df_products, suffix_manu,th_manu)
    df_input_filtered=create_fuzzy_match_columns_statistics(df_input_filtered, df_products, suffix_desc,th_desc)



    df_final=enhanced_breakdown_and_final_match(df_input_filtered, df_products)

    return df_final






if __name__ == "__main__":
    # df_MARA =get_table_AI('MARA_Products', 'AI')
    # df_med =get_table_AI('Med_data', 'AI')
    # df_update_price =get_table_AI(table_name='A501_A703_A503_Updated_prices',db_label='AI')
    # df_MARA.to_parquet(os.path.join(root_path, 'Data/MARA_Products.parquet'), index=False)
    # df_med.to_parquet(os.path.join(root_path, 'Data/Med_data.parquet'), index=False)
    # df_update_price.to_parquet(os.path.join(root_path, 'Data/A501_A703_A503_Updated_prices.parquet'), index=False)


    input_path = os.path.join(workspace_root, 'input')
    #file_name='Medical Commodities IMC - input'
    file_name='Sarel_TBL JER 2025 01 EN Health standard list - input'
    df_input = pd.read_excel(os.path.join(input_path, f'{file_name}.xlsx'))
    #df=read_prodcuts_data()
    df_final=create_match_product(df_input)



    
    # #region separete function
    # #region step 1 - read the products data
    # df_products=read_prodcuts_data()
    # # a=df_products[df_products['MATNR'].isin(['2093563518', '2093697964','2093563488','2093563519'])]

    # #endregion

    # #region step 2 - read the input data and filter it
    # #read input if we are working only in this file.
    # # input_path = os.path.join(root_path, 'input')
    # # df_test1 = pd.read_excel(os.path.join(input_path, 'test1.xlsx'))
    # # df_test2 = pd.read_excel(os.path.join(input_path, 'test2.xlsx'))
    # # df_input = pd.concat([df_test1, df_test2], ignore_index=True)
    # # df_input.to_excel(os.path.join(input_path, 'input1.xlsx'), index=False)

    # #endregion
    # print(f'The original number of input records {len(df_input)}')
    # # input_desc_norm='Description_norm'
    # df_input_filtered=filtered_input_data(df_input)


    # #region step 3 - apply fuzzy matching and updated df

    # #parames
    # # final_desc = 'desc_fix'
    # suffix_manu = "manu_"
    # suffix_desc = "desc_"
    # suffix_gen = "gen_"
    # # th_manu= 90
    # # th_desc = 85


    # # #test
    # # matches = process.extract(
    # # "BLOOD GROUPING SERUM ANTI- A, MONOCLONAL, 10ml, btl.",
    # # df_products[final_desc].dropna().unique().tolist(),
    # # scorer=fuzz.token_set_ratio, limit=len( df_products[final_desc].dropna().unique().tolist()))


    # df_input_manu=create_fuzzy_match_columns_statistics(df_input_filtered.copy(), df_products, suffix_manu,th_manu)
    # df_input_desc=create_fuzzy_match_columns_statistics(df_input_manu.copy(), df_products, suffix_desc,th_desc)
    # df_input_gen=create_fuzzy_match_columns_statistics(df_input_desc.copy(), df_products, suffix_gen,25)

    # def normalize_and_tokenize(text):
    #     """Lowercase, remove punctuation, and split into unique tokens."""
    #     if not isinstance(text, str):
    #         return set()
    #     text = re.sub(r"[^a-z0-9\s]", " ", text.lower())  # remove punctuation
    #     tokens = set(text.split())
    #     return tokens

    # def get_tokenwise_exact_generic_matches(input_val, reference_list, top_k=5):
    #     """
    #     Check if any generic names are contained in the input description string.
        
    #     Args:
    #         input_val (str): The input description to search within.
    #         reference_list (list): The list of generic names to look for.
    #         top_k (int): Number of top generic names to return based on length (optional).
            
    #     Returns:
    #         - to_match: The original input description
    #         - matched_generics: List of matching generic names (that are substrings)
    #         - best_match: The longest matching generic (optional heuristic)
    #         - match_score: Score (e.g., number of matched tokens or length of best match)
    #     Notes:
    #         Matching is case-insensitive and token-based (ignores partial tokens).
    #     """
    #     # Validate input
    #     if not isinstance(input_val, str) or input_val.strip() == "":
    #         return input_val, None, None, None
    #     input_tokens = normalize_and_tokenize(input_val)
    #     matched_generics = []
    #     # input_lower = input_val.lower()
    #     # matches = [g for g in reference_list if g.lower() in input_lower]
        
    #     # Sort matches by length (optional, longer = more specific)
    #     # sorted_matches = sorted(matches, key=lambda x: -len(x))
        
    #     # best_match = sorted_matches[0] if sorted_matches else None
    #     # match_score = 1

    #     for generic in reference_list:
    #         generic_tokens = normalize_and_tokenize(generic)
    #         if generic_tokens.issubset(input_tokens):
    #             matched_generics.append(generic)
    #      # Sort by length (you can change to score etc. if needed)
    #     matched_generics_sorted = sorted(matched_generics, key=lambda x: -len(x))
    #     best_match = matched_generics_sorted[0] if matched_generics_sorted else None
    #     score = len(matched_generics_sorted)

    #     return matched_generics_sorted[:top_k], best_match, score

    
    # reference_generics = (
    #     df_products[med_generic]
    #     .dropna()                      # Remove NaN values
    #     .loc[lambda x: x != '']       # Remove empty strings
    #     .unique()
    #     .tolist()
    # )

    # df_input_gen[["generic_matches", "generic_best_match", "generic_match_score"]] = df_input_gen[input_desc_norm].apply(
    #     lambda x: pd.Series(get_tokenwise_exact_generic_matches(x, reference_generics))
    # )
    # # df_input_gen.drop(columns='generic_to_match',inplace=True)



    # #df_final=enhanced_breakdown_and_final_match(df_input_gen.copy(), df_products)
    # #print(df_final.columns)
    # # df_final_filter=df_final[['input_id', 'Id number', 'Description', 'Category',
    # #    'Unit of Measure\n in smallest unit', 'Quantity \nin smallest unit',
    # #    'Unit Price in USD\nper smallest unit', 'Delivery Time\n in days',
    # #    'Alternative item / Comments',
    # #    'desc_top_matches','gen_top_matches','generic_best_match','generic_match_score']]
    # df_final_filter=df_input_gen[['input_id', 'Code', 'Description', 
    # 'desc_top_matches','gen_top_matches','generic_best_match','generic_match_score']]
    
    # def explode_and_normalize_dict_columns(df, dict_columns_with_prefix):
    #     """
    #     Explodes and normalizes dictionary columns from a DataFrame.

    #     Args:
    #         df (pd.DataFrame): Input DataFrame.
    #         dict_columns_with_prefix (dict): Dictionary where keys are column names 
    #                                         to process and values are their prefixes.
    #                                         Example: {'desc_top_matches': 'desc_', 'gen_top_matches': 'gen_'}

    #     Returns:
    #         pd.DataFrame: Transformed DataFrame with exploded and normalized columns.
    #     """
    #     df = df.copy()

    #     # Step 1: Explode all relevant columns
    #     df = df.explode(list(dict_columns_with_prefix.keys())).reset_index(drop=True)

    #     # Step 2: Normalize and prefix each exploded dict column
    #     normalized_parts = []
    #     for col, prefix in dict_columns_with_prefix.items():
    #         norm = pd.json_normalize(df[col])
    #         norm.columns = [f"{prefix}{c}" for c in norm.columns]
    #         normalized_parts.append(norm)

    #     # Step 3: Drop original dict columns and concatenate the normalized ones
    #     df = df.drop(columns=dict_columns_with_prefix.keys())
    #     df = pd.concat([df] + normalized_parts, axis=1)

    #     return df


    # dict_columns = {
    # 'desc_top_matches': 'desc_',
    # 'gen_top_matches': 'gen_'
    # }
    # df_exploded=explode_and_normalize_dict_columns(df_final_filter.copy(), dict_columns)


    # df_exploded['gen_match']=np.where(
    #     df_exploded['gen_score']<40,np.nan,df_exploded['gen_match']
    # )

    # df_exploded['generic_final']=df_exploded['generic_best_match'].fillna(df_exploded['gen_match'])
    # df_exploded['generic_score'] = np.where(
    #     df_exploded['generic_best_match'].isna(),df_exploded['gen_score'],90   
    # )
    # print(df_exploded.columns)
    # df_exploded.drop(columns=['generic_best_match','generic_match_score','gen_score','gen_match'],inplace=True)


    # ### for each row merge according to the final result
    # df_exploded_desc=df_exploded.merge(df_products[[final_desc,product_id_MARA,med_generic]], left_on='desc_match', right_on=final_desc, how='left')
    # df_products_med_filter=df_products[~df_products[med_generic].isna()]
    # df_exploded_gen=df_exploded_desc.merge(df_products_med_filter[[final_desc,product_id_MARA,med_generic]], left_on='generic_final', right_on=med_generic, how='left')

    # base_cols = [
    # 'input_id', 'Code', 'Description'
    # ]

    # df_x = df_exploded_gen[base_cols + ['MATNR_x', 'desc_fix_x','fix_Generic_Name_x','desc_score']]
    # df_x['source']='desc'
    # df_y = df_exploded_gen[base_cols + ['MATNR_y', 'desc_fix_y','fix_Generic_Name_y','generic_score']]
    # df_y['source']='generic'
    # df_x = df_x.rename(columns={'MATNR_x': 'MATNR', 'desc_fix_x': 'desc_fix','fix_Generic_Name_x':'fix_Generic_Name','desc_score':'score'})
    # df_y = df_y.rename(columns={'MATNR_y': 'MATNR', 'desc_fix_y': 'desc_fix','fix_Generic_Name_y':'fix_Generic_Name','generic_score':'score'})

    
    # # Step 2: Build a mapping from input_id → set of valid generic_final values from desc rows
    # desc_generics_per_input = (
    #     df_x.groupby(input_id)[med_generic]
    # .agg(lambda x: set(filter(pd.notna, x)))  # keep non-null only
    # .to_dict()
    # )

    
    # # Step 3: Define filter mask for generic rows to DROP
    # def should_drop(row):
    #     input_id_val = row[input_id]
    #     generic_val = row[med_generic]
    #     score = row['score']

    #     desc_generics = desc_generics_per_input.get(input_id_val, set())

    #     # If this generic value is not in desc and score < 50 → drop it
    #     if generic_val not in desc_generics and score <= 50:
    #         return True

    #     return False
    
    # # Step 4: Apply filter to generic rows
    # df_y_keep = df_y[~df_y.apply(should_drop, axis=1)]

    # df_flattened = pd.concat([df_x, df_y_keep], ignore_index=True)

    # df_flattened_filter=df_flattened[df_flattened[product_id_MARA].notna() & (df_flattened[product_id_MARA] != '')]
    # # df_flattened_filter.to_excel(os.path.join(input_path, 'test4.xlsx'), index=False)


    # df_flattened_filter = df_flattened_filter.sort_values(by='score', ascending=False)
    # # Step 2: Group and aggregate
    # df_flattened_final = (
    #     df_flattened_filter
    #     .groupby([input_id, product_id_MARA], as_index=False)
    #     .agg({
    #         'score': 'first',
    #         'source': lambda x: 'both' if len(set(x)) > 1 else list(x)[0],
    #         final_desc: 'first',
    #         med_generic: 'first',
    #         # Add more fields as needed
    #     })
        
    # )

    # df_flattened_final['score'] = df_flattened_final.apply(
    # lambda row: row['score'] + 10 if row['source'] == 'both' else row['score'],
    # axis=1
    # )

    # def filter_high_scores_only(group):
    #     scores = group['score']
    #     has_100 = (scores >= 100).any()
    #     has_90 = (scores >= 90).any()

    #     if has_100 and has_90:
    #         return group[scores >= 90]  # keep 90 and above
    #     elif has_90:
    #         return group[scores >= 62]
    #     else:
    #         return group

    # # Apply the logic
    # df_flattened_final = df_flattened_final.groupby(input_id, group_keys=False).apply(filter_high_scores_only)

    # df_flattened_final_result=df_input_filtered.merge(df_flattened_final, on=input_id,how='left')
    # df_final_result=df_flattened_final_result.merge(df_products[[product_id_MARA,product_dv_id,product_desc, manufacturer_model, supplier_name, med_concentration, med_dosage, med_dose,price_in_coin, price_coin, price_unit, price_unit_ILS]],on=product_id_MARA,how='left')
    # print(len(df_final_result))


    # def filter_and_boost_by_column(df, column_name):
    #     """
    #         Filters and boosts match scores within each group based on a column's value being present in the normalized input description.

    #         Parameters:
    #         -----------
    #         df : pd.DataFrame
    #             The DataFrame containing match results, including a score column and normalized input descriptions.

    #         column_name : str
    #             The name of the column to check (e.g., 'Generic', 'Substance', etc.).
    #             The function checks whether the column value (if not empty) appears in the normalized input description.

    #         Returns:
    #         --------
    #         pd.DataFrame
    #             The updated DataFrame. If a row's `column_name` value is found inside the normalized input description,
    #             its match score is increased by 10. This is done per group defined by `input_id`.

    #         Notes:
    #         ------
    #         - The function groups rows by `input_id` (must be defined globally).
    #         - Within each group, if any row has a non-empty `column_name` value that exists in the `input_desc_norm` field,
    #         only those rows are kept and their score is boosted by +10.
    #         - If no matches are found in a group, the group is returned unchanged.
    #         - This helps favor rows where important metadata (like a generic name) explicitly appears in the normalized input description.

    #     """
    #     def process_group(group):
    #         # Find rows where the column is not empty and appears in the Description
    #         mask = group[column_name].notna() & group[column_name].astype(str).str.strip().ne('') & group.apply(
    #             lambda row: str(row[column_name]).lower() in str(row[input_desc_norm]).lower(), axis=1
    #         )

    #         if mask.any():
    #             filtered = group[mask].copy()
    #             filtered['score'] += 10
    #             return filtered
    #         else:
    #             return group

    #     return df.groupby(input_id, group_keys=False).apply(process_group)
    
    # # First pass: Concentration
    # df_final_result_con = filter_and_boost_by_column(df_final_result.copy(), med_concentration)

    # # Second pass: Dosage
    # df_final_result_dosage = filter_and_boost_by_column(df_final_result_con.copy(), med_dose)

    # df_final_result_dosage['score'] = df_final_result_dosage['score'].replace({
    # 120: 97,
    # 110: 94,
    # 100: 92
    #     })
    # print(df_final_result_dosage[input_id].nunique())

    
    
    # # df_update_price_filter=df_update_price[df_update_price['HIENR']=='X']
    # # df_final_result_price=df_final_result_dosage.merge(df_update_price_filter[[product_id_MARA,'KBETR', 'KONWA', 'KMEIN', 'KMEIN_ILS']], on='MATNR', how='left')
    # df_final_result_dosage.sort_values(by=[input_id, 'score'], ascending=[True, False], inplace=True, ignore_index=True)
    # df_final_result_dosage['match_num'] = df_final_result_dosage.groupby(input_id)[product_id_MARA].transform('count')
    # df_final_result_dosage.drop(columns=[final_desc,input_desc_norm ,med_concentration,med_dosage,med_dose, med_generic],inplace=True)

    # #    'desc_matches_above_threshold', 'desc_best_match', 'desc_best_score',
    # #    'MFRPN', 'desc_product_match', 'desc_dv_match', 'final_match_product',
    # #    'final_match_dv', 'final_match_desc', 'final_match_manufacturer_model',
    # #    'final_match_score', 'final_match_source', 'rule_case']

    # df_exploded_merge_price_filter=df_final_result_dosage.drop_duplicates()
    # df_exploded_merge_price_filter.drop(columns=['match'],inplace=True, errors='ignore')
    # df_exploded_merge_price_filter.to_excel(os.path.join(input_path, file_name+' result3.xlsx'), index=False)

    
    
    # # a=df_final_up[df_final_up.duplicated(subset='input_id', keep=False)]



    # #endregion
    # #endregion


    # #region embed descriptions and find neighbors

    # import torch
    # from transformers import AutoTokenizer, AutoModel
    # from sklearn.preprocessing import normalize
    # from collections import Counter


    # #🔁 Step 1: Embed full descriptions
    # def embed_texts(texts, model, tokenizer, device, max_length=64):
    #     """
    #     Embed a list of full text descriptions using a BERT model.
    #     """
    #     model.eval()
    #     encoded = tokenizer(
    #         texts,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt"
    #     ).to(device)

    #     with torch.no_grad():
    #         outputs = model(**encoded)
        
    #     embeddings = outputs.last_hidden_state
    #     attention_mask = encoded["attention_mask"].unsqueeze(-1)
    #     summed = torch.sum(embeddings * attention_mask, dim=1)
    #     counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    #     mean_pooled = summed / counts
    #     return mean_pooled.cpu().numpy()


    # #🧰 Step 3: Full pipeline for model + description clustering
    # def compute_description_neighbors(df, text_col, model_name, model_label, top_k=5, threshold=0.9,df_ref=None,text_col_ref=None):
    #     """
    #     Compute semantic neighbors for product descriptions using a specified transformer model.
    #     This function encodes a text column using a transformer-based language model, then uses FAISS to find:
    #     - The top-K most similar descriptions
    #     - All descriptions above a similarity threshold

    #     These results are saved to two new columns:
    #     - 'top_desc_<model_label>': list of top-K closest descriptions
    #     - 'th_desc_<model_label>': list of descriptions with similarity above the threshold
    #     Args:
    #         df (pd.DataFrame): DataFrame with a text column to embed
    #         text_col (str): Name of the column containing text descriptions
    #         model_name (str): Hugging Face model name
    #         model_label (str): Label to differentiate model outputs in column names
    #         top_k (int): Number of top neighbors to retrieve (default is 5).
    #         threshold (float): Similarity threshold for neighbors (default is 0.9).
    #         df_ref (pd.DataFrame, optional): Reference DataFrame to compare against.
    #         text_col_ref (str, optional): Text column name in df_ref.
        
    #     Returns:
    #         pd.DataFrame: Original df with two new columns:
    #         - top_desc_<model_label>: list of top-K closest descriptions
    #         - th_desc_<model_label>: list of descriptions above the threshold
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = AutoModel.from_pretrained(model_name).to(device)
    #     model.eval()

    #     # Embed df texts
    #     texts = df[text_col].fillna("").tolist()
    #     vectors = []
    #     batch_size = 64 # Batch process to avoid overload
    #     for i in range(0, len(texts), batch_size):
    #         batch = texts[i:i+batch_size]
    #         vecs = embed_texts(batch, model, tokenizer, device)
    #         vectors.append(vecs)

    #     all_vectors = np.vstack(vectors)
    #     vectors_norm = normalize(all_vectors, axis=1)

    #     # If no reference df provided, compare df to itself (reuse vectors_norm)
    #     if df_ref is None or text_col_ref is None:
    #         ref_vectors_norm = vectors_norm
    #         ref_texts = texts
    #     else:
    #         # Embed reference df texts separately
    #         ref_texts = df_ref[text_col_ref].fillna("").tolist()
    #         ref_vectors = []
    #         for i in range(0, len(ref_texts), batch_size):
    #             batch = ref_texts[i : i + batch_size]
    #             vecs = embed_texts(batch, model, tokenizer, device)
    #             ref_vectors.append(vecs)
    #         ref_all_vectors = np.vstack(ref_vectors)
    #         ref_vectors_norm = normalize(ref_all_vectors, axis=1)

    #     # Save top-K and threshold neighbors
    #     # top_neighbors = get_faiss_neighbors(vectors_norm, texts, top_k=top_k)
    #     # th_neighbors = get_faiss_neighbors(vectors_norm, texts, threshold=threshold)
    #     top_neighbors = get_faiss_neighbors(ref_vectors_norm, ref_texts, top_k=top_k)
    #     th_neighbors = get_faiss_neighbors(ref_vectors_norm, ref_texts, threshold=threshold)


    #     df[f'top_desc_{model_label}'] = top_neighbors
    #     df[f'th_desc_{model_label}'] = th_neighbors
    #     return df

    # print(df_products.columns)
    # # Run for Bio_ClinicalBERT
    # df_filter_model1=compute_description_neighbors(
    #     df_products[[product_id_MARA,product_desc,final_desc]].copy(),
    #     text_col=final_desc,
    #     model_name="emilyalsentzer/Bio_ClinicalBERT",
    #     model_label="bio"
    # )

    # df_filter_model1.insert(
    #     df_filter_model1.columns.get_loc('top_desc_bio') + 1,
    #     'top_desc_bio_tokens',
    #     df_filter_model1['top_desc_bio'].apply(lambda desc_list: tokenize_text(" ".join(desc_list)) if isinstance(desc_list, list) else [])
    # )


    # # Run for ClinicalBERT
    # df_filter_model_2=compute_description_neighbors(
    #     df_filter[[product_id_MARA,product_desc, product_desc_update,rules_desc]].copy(),
    #     text_col=rules_desc,
    #     model_name="medicalai/ClinicalBERT",
    #     model_label="med"
    # )
    # df_filter_model_2.insert(
    #     df_filter_model_2.columns.get_loc('th_desc_med') + 1,
    #     'th_desc_med_tokens',
    #     df_filter_model_2['th_desc_med'].apply(lambda desc_list: tokenize_text(" ".join(desc_list)) if isinstance(desc_list, list) else [])
    # )



    # df_filter_model_2.insert(
    #     df_filter_model_2.columns.get_loc('th_desc_med') + 2,
    #     'th_desc_med_token_freq',
    #     df_filter_model_2['th_desc_med'].apply(
    #         lambda desc_list: dict(Counter(tokenize_text(" ".join(desc_list)))) 
    #         if isinstance(desc_list, list) else {}
    #     )
    # )

    # #endregion

