#region import libraries
import datetime
import os
import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter,defaultdict
import random
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.preprocessing import normalize
import openai
from openai import OpenAI
import time
import json
import ast

from dotenv import load_dotenv



# Add root directory to sys.path safely for both script and interactive environments
try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    print(f"Root path set to: {root_path}")

except NameError:
    # __file__ is not defined in Interactive Window
    root_path = os.path.abspath(os.path.join(os.getcwd(), '../../'))

if root_path not in sys.path:
    sys.path.append(root_path)
    print(f"Added root path to sys.path: {root_path}")

# Now safely import
# from database.db_AI_utils import load_dataframe_to_table, inserted_column, replace_empty_with_null_safe,get_table_AI
from database.db_AI_utils import *
from app.constants import *
from app.prompts import *
load_dotenv()



#endregion


#region functions

def filter_products(df_MARA, df_CE1SARL):

    """
    Filters the DataFrame to include only products:
    - from specified divisions.
    - from specific years
    - where the product ID starts with a digit.
    - without hebrew characters in the description.

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
        # Remove all occurrences of `manu` as a whole word
        #return desc.replace(manu,"").strip()
        return re.sub(rf'\b{re.escape(manu)}\b', '', desc).strip()
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

def tokenize_text(text):
    tokens= word_tokenize(text)
    # # Keep tokens that are at least one letter or number (skip pure punctuation)
    return [token.lower() for token in tokens if re.search(r'[a-zA-Z0-9]', token)]

def clean_space_and_special_chars(text):
    """
    Helper function to clean special characters from the end of a match.
    """
    #after all rules applied, remove any leading or trailing special characters and whitespace
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove trailing special characters and whitespace
    text = re.sub(r'[\s\-.,/\\]+$', '', text)
    return text.strip()

def preprocess_description(text, rules=None):
    """
    Generic, extensible text preprocessor.

    rules is a list containing any of:
      - "replace_punct": replace , / \\ - *  → " "
      - "smart_dot_split": replace "." → " " except in 2‐letter abbr’s like "W.O"
      - "remove_digits": remove any run of digits not sandwiched between letters
    after all the rule are applied: trim_trailing": strip trailing spaces and punctuation [-.,/\\]
    """

    if not isinstance(text, str):
        return ""

    if rules is None:
        rules = [
            "replace_punct",
            "smart_dot_split",
            "remove_digits",
        ]
    
    # Rule 1: Replace commas and slashes, - ,* with space
    if "replace_punct" in rules:
        text = re.sub(r'[,/\\\-\*\#\%\(\)\+\=]', ' ', text)

    # Rule 2: Smart dot split (preserve abbreviations like "W.O", break others like "w.cath")
    if "smart_dot_split" in rules:
        def dot_replacer(match):
            part = match.group()
            parts = part.split('.')
            if all(p.isalpha() and len(p) <= 2 for p in parts):
                return part  # Keep as abbreviation (e.g., F.A.M.A, W.O)
            else:
                return ' '.join(parts)  # Break apart (e.g., lens.intra → lens intra)

        # Match any sequence with one or more dot-separated words/numbers
        text = re.sub(r'\b(?:\w+\.)+\w+\b', dot_replacer, text)

    # Rule 3: Remove digits
    if "remove_digits" in rules:
        # Match digits that are NOT surrounded by letters on both sides
        # Remove floats or integers not between letters
        #text = re.sub(r'(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])', '', text)
        text = re.sub(r'\d+', '', text)
    
    # Apply the cleaning function to the entire text
    text_final = clean_space_and_special_chars(text)
    return text_final

def embed_tokens(tokens_list, model, tokenizer, device):
    """    
        Generate embeddings for a list of tokens or short phrases using a pretrained BERT model.
        Steps:
    1. Wrap each token/phrase in a list to treat them as individual 'sentences'.
    2. Tokenize using the Hugging Face tokenizer with padding and truncation.
    3. Feed the encoded inputs to the BERT model without computing gradients (inference mode).
    4. Extract the last hidden layer (embeddings) from the model output.
    5. Perform mean pooling over the embeddings, excluding [PAD] tokens using the attention mask.
    6. Return the final mean-pooled embeddings as NumPy arrays.
    Parameters:
        tokens_list (list of str): Tokens or phrases to embed.

    Returns:
        numpy.ndarray: Array of shape (len(tokens_list), hidden_size), e.g., (batch_size, 768)

    """
    model.eval()  # Ensure model is in eval mode

    # Prepare tokens as list of single-token sentences
    batch = [[token] for token in tokens_list]
    # Tokenize each token (or phrase)
    encoded = tokenizer(batch,
                        is_split_into_words=True,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=32).to(device)
    
    with torch.no_grad():  # Disable gradients to save memory
        outputs = model(**encoded)

    embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

    # Use attention mask to mean-pool only real tokens (not [PAD])
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    # Zero out [PAD] token embeddings
    summed = torch.sum(embeddings * attention_mask, dim=1)
    # Number of real tokens per input
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts     # Final vector: mean over non-pad tokens
    return mean_pooled.cpu().numpy()  # Shape: (batch_size, 768)

def get_faiss_neighbors(vectors, tokens, top_k=None, threshold=None):
    """
    Find similar tokens using FAISS with either top-k or similarity threshold filtering.
    
    Args:
        vectors (np.ndarray): 2D array of shape (n_tokens, 768), already normalized for cosine similarity.
        tokens (List[str]): List of token strings corresponding to the vectors.
        top_k (int, optional): Number of top neighbors to retrieve (excluding self). Default: None.
        threshold (float, optional): Cosine similarity threshold (e.g., 0.85). Default: None.

    Returns:
        List[List[str]]: List of neighbor lists for each token.
    """

    # Create FAISS index (cosine similarity via inner product on unit vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Decide max number to retrieve
    search_k = top_k + 1 if top_k is not None else 50  # 50 is safe default for threshold mode
    D, I = index.search(vectors, search_k)

    neighbors_list = []

    for i, (sim_scores, idx_list) in enumerate(zip(D, I)):
        neighbors = []

        for score, idx in zip(sim_scores[1:], idx_list[1:]):  # skip self
            if threshold is not None and score < threshold:
                continue
            neighbors.append(tokens[idx])
            if top_k is not None and len(neighbors) >= top_k:
                break

        neighbors_list.append(neighbors)

    return neighbors_list

def is_number(token):
    # Check if a token is a number (integer or float)
    return bool(re.fullmatch(r'-?\d+(\.\d+)?', token))


def replace_shortcuts_token_based(text, shortcut_dict,blocked_shortcuts=None):
    """
    Replace only full-word shortcuts using smart token-based scanning.
    Prefer longer shortcuts over shorter ones (e.g., 'w.o' > 'w.' > 'w').

    Returns:
        Tuple[str, List[str]]: (updated text, list of used shortcuts)
    """
    if not isinstance(text, str):
        return text, []

    text_lower = text.lower()
    used_shortcuts = []
    blocked_shortcuts = blocked_shortcuts or {}


    # Sort shortcuts by length descending, so longer ones are matched first
    sorted_shortcuts = sorted(shortcut_dict.items(), key=lambda x: -len(x[0]))

    # Split text into tokens keeping punctuation (so "w.blood" => ["w", ".", "blood"])
    tokens = re.findall(r'\w+|[^\w\s]', text_lower)

    i = 0
    output_tokens = []
    while i < len(tokens):
        matched = False

        for shortcut, full_word in sorted_shortcuts:
            shortcut_tokens = re.findall(r'\w+|[^\w\s]', shortcut.lower())
            length = len(shortcut_tokens)

            # Check for block: if this sequence is in blocked_shortcuts, skip it
            if blocked_shortcuts is not None:
                candidate_tokens = tokens[i:i+length]
                candidate = ''.join(candidate_tokens)
                # Only block if exact match and not handled by replacement dict
                if candidate in blocked_shortcuts and candidate not in shortcut_dict:
                    matched = True
                    i += length
                    output_tokens.extend(candidate_tokens)
                    break

            # Replace if exact match found in shortcut_dict
            if tokens[i:i+length] == shortcut_tokens:
                output_tokens.append(full_word)
                used_shortcuts.append(shortcut)
                i += length
                matched = True
                break

        if not matched:
            output_tokens.append(tokens[i])
            i += 1

    # Join the tokens with proper spacing
    result = ""
    for j, tok in enumerate(output_tokens):
        if j > 0 and re.match(r'\w', tok) and re.match(r'\w', output_tokens[j-1]):
            result += " "
        result += tok

    return result, list(set(used_shortcuts))

def build_user_message(batch):
    """
    Build the user message for OpenAI API from a batch of shortcuts and their example descriptions.
    Each shortcut will be presented with its examples for analysis.
    The message will be formatted to guide the model in interpreting the shortcuts.
    :param batch: DataFrame containing shortcuts and their example descriptions.
    :return: JSON Formatted message string for OpenAI API.
    """

    message = user_message_abbreviations
    for _, row in batch.iterrows():
        shortcut = row['token']
        examples = row['examples']
        message += f"\nShortcut: \"{shortcut}\"\n"
        for i, desc in enumerate(examples):
            message += f"{i}. {desc}\n"
    return message


def call_openai(system_message, user_message, max_token=2000, retries=3, delay=5):
    """
    Call OpenAI API with retries and delay.
    :param system_message: The system prompt to guide the model. 
    :param user_message: The user message containing the shortcuts and examples.
    :param max_token: Maximum tokens for the response.
    :param retries: Number of retries in case of failure.
    :param delay: Delay in seconds between retries.
    :return: The response from OpenAI API or None if all retries fail.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name_abbreviations,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                max_tokens=max_token
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    return None

def flatten_alternatives(row, max_alternatives=5):
    """
    Flatten the 'gpt_meaning_alternatives' field into separate columns.
    Each alternative will have its own set of columns for meaning, desc_match, and accuracy.
    :param row: A row from the DataFrame containing 'gpt_meaning_alternatives'.
    :param max_alternatives: Maximum number of alternatives to flatten (default is 5).
    :return: A Series with flattened alternative fields."""
    alt_data = {}
    alternatives = row.get("gpt_meaning_alternatives", [])
    if not isinstance(alternatives, list):
        return pd.Series(alt_data)  # Return empty if it's not a list
    
    for i, alt in enumerate(alternatives[:max_alternatives], 1):
        alt_data[f"gpt_meaning_other_{i}"] = alt.get("meaning")
        alt_data[f"gpt_desc_match_other_{i}"] = alt.get("desc_match")
        alt_data[f"gpt_accuracy_other_{i}"] = alt.get("accuracy")
        
    return pd.Series(alt_data)

def create_df_from_prompt_results(results, df_input):
    f"""
    Create a DataFrame from the results of OpenAI API calls.
    Each result should be a JSON string representing a list of dictionaries.
    If a result is not a valid JSON, it will be skipped and an error message will be printed.
    :param results: List of JSON strings returned by OpenAI API.
    :return: DataFrame containing the parsed results.
    columns: {input_shortcut},{gpt_meaning},{gpt_desc_match},{gpt_accuracy},{gpt_meaning_alternatives}, {reasoning}
    """
    parsed_rows = []
    for res_text in results:
        try:
            # Some responses may contain multiple JSON objects → wrap in []
            if not res_text.strip().startswith('['):
                res_text = "[" + res_text + "]"

            parsed = json.loads(res_text)
            # Handle both single and multiple JSON blocks
            if isinstance(parsed, dict):
                parsed_rows.append(parsed)
            elif isinstance(parsed, list):
                parsed_rows.extend(parsed)
        except json.JSONDecodeError as e:
            print("Failed to parse:", res_text)
            print("Error:", e)
    df_JSON=pd.DataFrame(parsed_rows)
    df_result = df_JSON.merge(df_input[[input_shortcut, 'examples']], on=input_shortcut, how='left')

    return df_result

def count_len_indices(indices):
    """
    Count the number of valid integer indices in the input.

    Parameters:
    -----------
    indices : str | list | pandas.Series (from prompt results)
        Index input, which can be a comma-separated string, list, or Series.

    Returns:
    --------
    int
        Number of valid integer indices.
    """
    if isinstance(indices, str):
        return len([i for i in indices.split(',') if i.strip().isdigit()])
    elif isinstance(indices, pd.Series):
        indices = indices.tolist()
    if isinstance(indices, list):
        return len([i for i in indices if isinstance(i, int)])
    return 0


def extract_match_descriptions(desc_list, indices):
    """
    Extract descriptions from a list using provided indices.

    Parameters:
    ----------
    desc_list : list of str
        A list of example descriptions (e.g., from the 'examples' column).
        
    indices : str | list | pandas.Series
        Indices that indicate which descriptions to extract. Can be:
        - A comma-separated string (e.g., '0,2,4')
        - A list of integers (e.g., [0, 2, 4])
        - A Series containing a list (e.g., pd.Series([0, 2, 4]))

    Returns:
    -------
    list of str
        A list of descriptions selected by index. Invalid entries are skipped.

    """
    if not isinstance(desc_list, list):
        return []

    # Convert comma-separated string to list of ints
    if isinstance(indices, str):
        try:
            indices = [int(i) for i in indices.split(',') if i.strip().isdigit()]
        except Exception:
            return []

    # Convert pandas Series to list
    if isinstance(indices, pd.Series):
        indices = indices.tolist()

    # Ensure we have a list of integers
    if not isinstance(indices, list):
        return []

    try:
        return [desc_list[i] for i in indices if isinstance(i, int) and 0 <= i < len(desc_list)]
    except Exception:
        return []

def create_normalized_df(df_prompt_merge, i_max):
    """
    Normalize the DataFrame by expanding each row into multiple rows based on the meanings and alternatives.
    Each row will contain the main meaning and any alternative meanings, along with their associated data.
    :param df_prompt_merge: DataFrame containing the prompt results with meanings and alternatives.
    :param i_max: Maximum number of alternative meanings columns to consider.
    :return: Normalized DataFrame with each meaning and alternative in a separate row.
    """
    # Initialize an empty list to hold the normalized rows
    normalized_rows = []

    # Loop through each row in your existing df
    for _, row in df_prompt_merge.iterrows():
        base = {
            "input_shortcut": row.get("input_shortcut"),
        }

        # Main meaning
        normalized_rows.append({
            **base,
            "meaning": row.get("gpt_meaning"),
            "desc_match": row.get("gpt_desc_match"),
            "matched_desc": row.get("matched_desc"),
            "desc_num": row.get("desc_num"),
            "accuracy": row.get("gpt_accuracy"),
            "from": "orginal",
        })

        # All alternatives (if any)
        for i in range(1, i_max+1):  # you can adjust max as needed
            if f"gpt_meaning_other_{i}" in row and pd.notna(row[f"gpt_meaning_other_{i}"]):
                normalized_rows.append({
                    **base,
                    "meaning": row.get(f"gpt_meaning_other_{i}"),
                    "desc_match": row.get(f"gpt_desc_match_other_{i}"),
                    "matched_desc": row.get(f"matched_desc_other_{i}"),
                    "desc_num": row.get(f"desc_num_other_{i}"),
                    "accuracy": row.get(f"gpt_accuracy_other_{i}"),
                    "from": f"other_{i}",
                })

    # Final tidy DataFrame
    df_normalized_meanings = pd.DataFrame(normalized_rows)
    return df_normalized_meanings.sort_values(by=["input_shortcut"], inplace=True,ignore_index=True)

#endregion

if __name__ == "__main__":

    #region Step init
    current_year=datetime.datetime.now().year
    years=np.arange(2023,current_year+1)

    print("Reading tables from DBAI...")
    df_CE1SARL = get_table_AI('CE1SARL_Invoice_all','AI', [], years)
    df_MARA =get_table_AI('MARA_Products')
    df_T023T=get_table_AI('T023T_MATKL_description')
    df_TSPAT=get_table_AI('TSPAT_dv_description')
    df_codes =get_table_AI('T006A_ZTMM035_ZTSD044_Code_units')
    df_family=get_table_AI('ZSD_MTL_FAMREL_V_Family')

    print("Tables read successfully.")



    df_MARA_fields=add_MARA_products_fields(df_MARA, df_T023T, df_TSPAT,df_codes)
    df_updated_products = filter_products(df_MARA_fields, df_CE1SARL)


    # random_sample = df_updated_products[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]].sample(n=100, random_state=42)
    # random_sample.to_excel('temp_excel/random_sample.xlsx', index=False)

    #endregion


    #region step1+step2: Preprocess descriptions and create vocabulary
    #step 1 - Preprocessing descriptions based on roles and create tokens
    df_filter=df_updated_products[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]]
    df_filter.insert(df_filter.columns.get_loc(product_desc_update) + 1, rules_desc, df_filter[product_desc_update].apply(preprocess_description))
    df_filter.insert(df_filter.columns.get_loc(rules_desc) + 1, 'tokens', df_filter[rules_desc].apply(tokenize_text))

    #Step 2: create a vocabulary of tokens with their frequencies
    # Step 2.1: Build vocab counter
    vocab_counter = Counter(token for tokens in df_filter["tokens"] for token in tokens)
    # Step 2.2: Build token → descriptions mapping once (very fast)
    token_to_desc = defaultdict(list)

    for tokens, desc in zip(df_filter["tokens"], df_filter[product_desc_update]):
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if pd.notna(desc):
                token_to_desc[token].append(desc)

    # Step 2.3: Create the vocab DataFrame
    vocab_records = []
    for token, freq in vocab_counter.items():
        all_descs = token_to_desc[token]
        examples = random.sample(all_descs, min(10, len(all_descs))) if all_descs else []
        vocab_records.append({
            "token": token,
            "frequency": freq,
            "examples": examples
        })

    df_vocab = pd.DataFrame(vocab_records).sort_values(by="frequency", ascending=False).reset_index(drop=True)
    df_vocab['examples_num']= df_vocab['examples'].apply(lambda x: len(x)) #add the number of examples for each token
    print(f"Vocabulary created with {len(df_vocab)} unique tokens.")

    #step 2.4: create filter vocabulary
    # Define a list of stopwords to remove (you can expand this list)
    # custom_stopwords = {'for', 'in', 'with', 'and', 'or', 'on', 'of', 'the', 'a', 'an'}
    custom_stopwords = {'and', 'or', 'on', 'of', 'the', 'a', 'an'}

    # Filter out according to frequency, numeric tokens and stopwords
    df_vocab_filter = df_vocab[
        (df_vocab["frequency"] >= 2) &  # keep tokens with frequency >= 2
        (~df_vocab['token'].apply(is_number)) &  # remove tokens that are only digits
        (~df_vocab['token'].isin(custom_stopwords))  # remove unwanted common words
    ].reset_index(drop=True)
    print(f"Vocabulary filter created with {len(df_vocab_filter)} unique tokens.")


    df_vocab_shourtcuts = df_vocab_filter[(df_vocab_filter['token'].str.contains('\.')) &
                                        (~df_vocab_filter['token'].str.contains(r'\d'))].reset_index(drop=True)
    print(f"Vocabulary shourtcuts created with {len(df_vocab_shourtcuts)} unique tokens.")
    print(f'All the shourtcuts are: {df_vocab_shourtcuts["token"].tolist()}')



    #endregion

    #region step 3: embeding tokens,Semantic Mapping & Abbreviation Expansion
    # MODEL = "emilyalsentzer/Bio_ClinicalBERT"  # or "medicalai/ClinicalBERT"
    MODEL = "medicalai/ClinicalBERT"  # or "medicalai/ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokens = df_vocab['token'].tolist()

    batch_size = 64
    vectors = []

    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        vecs = embed_tokens(batch, model, tokenizer, device)
        vectors.append(vecs)

    all_vectors = np.vstack(vectors)  # <- Combine all batches
    df_vocab["vector"] = list(all_vectors) #vector for each token for similarity search

    #find similar token to the shourtcuts
    #Top-k (e.g., top 5 neighbors)
    vectors_norm = normalize(all_vectors, axis=1)
    top_neighbors = get_faiss_neighbors(vectors_norm, df_vocab['token'].tolist(), top_k=5)
    df_vocab['top_neighbors'] = top_neighbors

    #Threshold-based 
    top_neighbors = get_faiss_neighbors(vectors_norm, df_vocab['token'].tolist(), threshold=0.93)
    df_vocab['th_neighbors'] = top_neighbors

    #endregion


    #region step 4: create dictonary with elboreated tokens and all the shourtcuts

    # df_manual_vocab = pd.DataFrame({
    #         "final_word": ['with','without','out', 'needle', 'catheter','disposable','for', 'scissor', 'glove'],
    #         "shourtcuts": [ ['w.','w'],  ['w.o'] ,['o','o.'],['ndl'],['cath'] ,['disp'], ['f.','f'], ['scs'], ['glv'] ]
    #         })

    #step 4.1 - manual vocabulary
    df_vocab_file=pd.read_excel('temp_result/medical_shortcuts.xlsx')


    #step 4.2: create a dictionary with elboreated tokens and all the shourtcuts using prompt engineering
    #region prompting for find meaning to the shortcuts
    new_openAI_key=os.getenv("new_openAI_key")
    client = OpenAI(api_key=new_openAI_key)  # Replace with your ke

   
    # Run in batches
    batch_size = 10
    results = []

    for start in range(0, len(df_vocab_shourtcuts), batch_size):
        # start=10
        batch = df_vocab_shourtcuts.iloc[start:start + batch_size]
        user_msg = build_user_message(batch)
        # Print the entire user_msg without truncation
        response = call_openai(system_message_abbreviations, user_msg)
        if response:
            results.append(response)
        else:
            print(f"Failed to process batch starting at {start}")

    # Create DataFrame from the JSON results (as is)+examples input
    df_result = create_df_from_prompt_results(results,df_vocab_shourtcuts)

    # df_result1 = pd.read_excel('temp_result/run1/df_prompt_result_org_1007.xlsx')
    # df_result2 = pd.read_excel('temp_result/run1/df_prompt_merge_160.xlsx')
    
    # df_result = df_result1.merge(df_result2[['input_shortcut', 'examples']], on='input_shortcut', how='left')
    
    

    def parse_list_string_from_prompt(val):
        # Function to safely parse JSON strings into Python objects
        if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "[]":
            return []
        try:
            # Safely evaluate the string into a Python object
            return ast.literal_eval(val)
        except Exception:
            return []
    
    df_result["gpt_meaning_alternatives_fix"] = df_result["gpt_meaning_alternatives"].apply(parse_list_string_from_prompt)
    df_result["gpt_desc_match_fix"] = df_result["gpt_desc_match"].apply(parse_list_string_from_prompt)
    df_result["examples_fix"] = df_result["examples"].apply(parse_list_string_from_prompt)
    df_result=df_result.rename(columns={"gpt_meaning_alternatives": "gpt_meaning_alternatives_str","gpt_meaning_alternatives_fix": "gpt_meaning_alternatives"})
    df_result=df_result.rename(columns={"gpt_desc_match": "gpt_desc_match_str","gpt_desc_match_fix": "gpt_desc_match"})
    df_result=df_result.rename(columns={"examples": "examples_str","examples_fix": "input_examples"})

    df_to_db=df_result[['input_shortcut', 'input_examples','gpt_meaning', 'gpt_desc_match', 'gpt_accuracy',
       'gpt_meaning_alternatives','reasoning']]
    df_to_db[prompt_date]=pd.to_datetime("2025-07-10")
    df_to_db[inserted_date]=datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    def to_json_for_sql(v):
        # treat pandas NaN/None as SQL NULL
        # if pd.isna(v):
        #     return None
        # lists/dicts -> JSON string
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        # already a string (maybe JSON) -> keep as-is
        if isinstance(v, str):
            return v
        # fallback: jsonify other scalar types (int/float -> "5", etc.)
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return json.dumps(str(v), ensure_ascii=False)

    cols_to_json = ['input_examples','gpt_desc_match', 'gpt_meaning_alternatives']
    for c in cols_to_json:
        print(df_to_db[c].map(type).unique())
        df_to_db[c] = df_to_db[c].apply(to_json_for_sql)


    load_dataframe_to_table(df=df_to_db, db_label='CATALOG_PLUS',table_name=prompt_table, mode='append')
    df_result_sql =get_table_AI(table_name=prompt_table, db_label='CATALOG_PLUS')


    def parse_json_or_fallback(s):
        if s is None:
            return None
        s = s.strip()
        # Quick test: if it looks like JSON array/object, try to load
        if s.startswith('[') or s.startswith('{'):
            try:
                return json.loads(s)
            except Exception:
                pass
        # fallback: maybe it's a comma-separated string (legacy); return list of tokens
        if ',' in s:
            return [t.strip() for t in s.split(',') if t.strip() != '']
        # otherwise return the raw string
        return s

    for c in cols_to_json:
        if c in df_result_sql.columns:
            df_result_sql[c] = df_result_sql[c].apply(parse_json_or_fallback)
            print(df_result_sql[c].map(type).unique())



    # Apply flattening
    df_flat_alts = df_result.apply(flatten_alternatives, axis=1)


    # Explicitly reorder columns by meaning1, match1, accuracy1, etc.
    ordered_cols = []
    # Extract the numeric suffix from the last column name and convert it to int
    i_max = int(df_flat_alts.columns[-1].split('_')[-1])
    for i in range(1, i_max+1):  # adjust max as needed
        print(i)
        ordered_cols += [
            f"gpt_meaning_other_{i}",
            f"gpt_desc_match_other_{i}",
            f"gpt_accuracy_other_{i}"
        ]

    # Reorder if all columns exist
    df_flat = df_flat_alts[[col for col in ordered_cols if col in df_flat_alts.columns]]
    # Merge with the original df_result
    df_result_expanded = pd.concat([df_result.drop(columns=["gpt_meaning_alternatives"]), df_flat], axis=1)
    df_result_DB = pd.concat([df_result, df_flat], axis=1)
    #df_result_DB.to_excel('temp_result/df_prompt_result_org_1007.xlsx', index=False)
    # df_result_expanded=pd.read_excel('temp_result/df_prompt_result_org_1007.xlsx')


    desc_cols = [col for col in df_result_expanded.columns if col.startswith("gpt_desc_match")]

    for col in desc_cols:
        count_col = col.replace("gpt_desc_match", "gpt_desc_num")
        insert_loc = df_result_expanded.columns.get_loc(col) + 1
        df_result_expanded.insert(insert_loc, count_col, df_result_expanded[col].apply(count_len_indices))

    #df_result_expanded.insert(df_result_expanded.columns.get_loc('gpt_desc_match') + 1, 'desc_num', df_result['gpt_desc_match'].apply(count_len_indices))
    # df_result.insert(df_result.columns.get_loc('gpt_desc_match_other') + 1, 'desc_num_other', df_result['gpt_desc_match_other'].apply(count_len_indices))


    #the goal is to check if all the desc_num and desc_num_other are equal to the examples_num
    df_prompt_merge = df_result_expanded.merge(df_vocab_shourtcuts[['token','examples', 'examples_num']], left_on='input_shortcut',right_on='token', how='left')
    df_prompt_merge.drop(columns=['token'], inplace=True)
    print(df_prompt_merge.columns)
    desc_other_cols = [col for col in df_prompt_merge.columns if col.startswith("desc_num_other")]
    df_prompt_merge['total_desc_prompt_match'] = df_prompt_merge['desc_num'] + df_prompt_merge[desc_other_cols].sum(axis=1)
    df_prompt_merge['vaild_desc_match'] = df_prompt_merge['desc_num']+df_prompt_merge[desc_other_cols].sum(axis=1)==df_prompt_merge['examples_num']
    print(df_prompt_merge.columns)

    # for each description column
    for col in desc_cols:
        new_col = col.replace("gpt_desc_match", "matched_desc")
        insert_loc = df_prompt_merge.columns.get_loc(col) + 1
        df_prompt_merge.insert(insert_loc, new_col, df_prompt_merge.apply(
            lambda row: extract_match_descriptions(row['examples'], row.get(col, '')), axis=1))
    # Save the final DataFrame to Excel
    df_prompt_merge.to_excel('df_prompt_merge_160.xlsx', index=False)

    # df_prompt_merge=pd.read_excel('temp_result/run1/df_prompt_merge_160.xlsx')
    # i_max=3
    #check if there are any meanings that are the same
    df_normalized_meanings=create_normalized_df(df_prompt_merge, i_max)
    df_same_meaning = df_normalized_meanings[df_normalized_meanings.duplicated(subset=['meaning'], keep=False)].sort_values(by='meaning', ignore_index=True)


    #filter the result only to the rows that the accuracy is above 0.85, only one meaning, the example number is above 5, and the desc_num is equal to the examples_num (vaild_desc_match)
    df_prompt_final=df_prompt_merge[(df_prompt_merge['gpt_meaning_other_1'].isna()) &
                            (df_prompt_merge['gpt_accuracy'] >= 0.9) & (df_prompt_merge['examples_num']>4) & (df_prompt_merge['vaild_desc_match'])].reset_index(drop=True)      

    df_prompt_final=df_prompt_final[['input_shortcut', 'gpt_meaning','gpt_accuracy','matched_desc', 'desc_num']]
    df_prompt_final.columns=['shortcut', 'meaning', 'accuracy', 'matched_desc', 'desc_num']
    df_prompt_final['source']='prompt'

    #endregion

    #step 4.3: merge the manual vocabulary with the prompt result
    df_final = pd.concat([df_vocab_file, df_prompt_final], ignore_index=True)
    print(f"Final vocabulary created with {len(df_final)} unique shortcuts.")
    print(f"Vocabulary prompt created with {len(df_prompt_final)} unique shortcuts.")

    #df_final.to_excel('df_final_voacb_prompt_manual.xlsx', index=False)
    df_final=pd.read_excel('temp_result/run1/df_final_voacb_prompt_manual.xlsx')
    df_final.dropna(subset=[col_shortcut, col_meaning], inplace=True)  # Ensure no NaN values in shortcuts or meanings
    
    #insert to the DB
    # col_shortcut='shortcut'
    # col_meaning='meaning'
    # col_source='source'
    # col_accuracy='accuracy'

    df_shortcut_table=df_final[[col_shortcut, col_meaning, col_source, col_accuracy]]
    df_shortcut_table[inserted_date]=datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    df_shortcut_table[updated_date]=datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    load_dataframe_to_table(df_shortcut_table, db_label='CATALOG_PLUS', table_name=shortcut_table, mode='append')




    # df_shortcut_stay = df_vocab[
    #     ~df_vocab['token'].str.lower().isin(df_final['shortcut'].str.lower())
    # ].reset_index(drop=True)[['token']]
    # df_shortcut_stay.rename(columns={'token': 'shortcut'}, inplace=True)
    # df_shortcut_stay['meaning'] = df_shortcut_stay['shortcut']  # Use the shortcut itself as meaning
    # df_shortcut_stay['source'] = 'still shortcut'  # Use the shortcut itself as meaning


    #endregion


    # region Step 5: apply the dictionary on the dscription column
    # Ensure lowercase for consistency and drop missing
    shortcut_dict = (
        df_final.dropna(subset=["shortcut", "meaning"])
                .assign(shortcut=lambda df: df["shortcut"].str.lower(),
                        meaning=lambda df: df["meaning"].str.lower())
                .set_index("shortcut")["meaning"]
                .to_dict()
    )
    # shortcut_block_dict = (
    #     df_shortcut_stay.dropna(subset=["shortcut", "meaning"])
    #             .assign(shortcut=lambda df: df["shortcut"].str.lower(),
    #                     meaning=lambda df: df["meaning"].str.lower())
    #             .set_index("shortcut")["meaning"]
    #             .to_dict()
    # )


    # Step 5.2: Function to replace all shortcuts in a description
    #df_filter.drop(columns='desc_fix', inplace=True)  # Remove existing 'desc_fix' column if it exists

    ##diffrent tries.
    # desc = "f.a.m.e MIX GLC-50 100MG NEAT"
    # desc="syring w.o.blood"
    # new_text, used = replace_shortcuts_token_based(desc.lower(), shortcut_dict,shortcut_block_dict)
    # print(new_text, used)

    df_filter[['desc_fix', 'shortcuts_used']] = df_filter[product_desc_update].apply(
        lambda x: pd.Series(replace_shortcuts_token_based(x, shortcut_dict))
    )


    # df_filter.insert(df_filter.columns.get_loc(product_desc_update) + 1, 'desc_fix', df_filter[product_desc_update].apply(lambda x: replace_shortcuts(x,shortcut_dict)))
    df_changes = df_filter[
        df_filter['shortcuts_used'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].reset_index(drop=True)[[product_id_MARA, product_desc_update, 'desc_fix', 'shortcuts_used']]

    # Flatten all used shortcuts into one set
    used_shortcuts = set(
        shortcut
        for used in df_filter['shortcuts_used']
        if isinstance(used, list)
        for shortcut in used
    )
    all_shortcuts = set(shortcut_dict.keys())
    unused_shortcuts = all_shortcuts - used_shortcuts
    unused_meanings = {shortcut_dict[sc] for sc in unused_shortcuts}
    # Print unused shortcuts and their meanings
    print(f"Unused shortcuts: {unused_shortcuts}")
    print(f"Unused meanings: {unused_meanings}")  


    df_new_desc=df_filter[[product_id_MARA,'desc_fix']]
    df_new_desc.to_excel('df_new_desc.xlsx', index=False)
    df_filter.to_excel('temp_result/df_new_desc_all_columns.xlsx', index=False)
    print(df_filter.columns)



    #endregion

    # region Step 6: Save the final vocabulary and changes to DB

 
    #next run test

    df_save=df_filter[[product_id_MARA, product_desc, product_desc_update, final_desc, manufacturer_model, 'shortcuts_used']]
    df_save_updated=inserted_column(replace_empty_with_null_safe(df=df_save.copy(), ls_drop=['shortcuts_used']),flag_insert=True, flag_update=False)
    df_save_updated['shortcuts_used']=df_save_updated['shortcuts_used'].apply(lambda x: json.dumps(x))
    # df_save_updated['shortcuts_used_JSON_text']=df_save_updated['shortcuts_used_JSON'].apply(lambda x: json.loads(x))
    load_dataframe_to_table(df_save_updated,'CATALOG_PLUS' ,products_desc_table, mode='replace')



    #endregion












































