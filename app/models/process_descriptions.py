import datetime
import os
import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict


# # Add the root directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from database.db_AI_utils import *

# Add root directory to sys.path safely for both script and interactive environments
try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
except NameError:
    # __file__ is not defined in Interactive Window
    root_path = os.path.abspath(os.path.join(os.getcwd(), '../../'))

if root_path not in sys.path:
    sys.path.append(root_path)

# Now safely import
from database.db_AI_utils import *


#enviroment variables
product_id_CE1SARL='ARTNR'

product_id_MARA='MATNR'
product_desc='full_desc'
product_desc_update='full_desc_update'
rules_desc='rules_desc'
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

#region Step init
current_year=datetime.datetime.now().year
years=np.arange(2023,current_year+1)

print("Reading tables from DBAI...")
df_CE1SARL = get_table_AI('CE1SARL_Invoice_all', [], years)
df_MARA =get_table_AI('MARA_Products')
df_T023T=get_table_AI('T023T_MATKL_description')
df_TSPAT=get_table_AI('TSPAT_dv_description')
df_codes =get_table_AI('T006A_ZTMM035_ZTSD044_Code_units')
print("Tables read successfully.")



df_MARA_fields=add_MARA_products_fields(df_MARA, df_T023T, df_TSPAT,df_codes)
df_updated_products = filter_products(df_MARA_fields, df_CE1SARL)


# random_sample = df_updated_products[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]].sample(n=100, random_state=42)
# random_sample.to_excel('temp_excel/random_sample.xlsx', index=False)

#endregion

#region step 1 - Preprocessing descriptions based on roles.



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
        text = re.sub(r'[,/\\\-\*\#\%\(\)\+]', ' ', text)

    # Rule 2: Smart dot split (preserve abbreviations like "W.O", break others like "w.cath")
    if "smart_dot_split" in rules:
        def dot_replacer(match):
            part = match.group()
            parts = part.split('.')
            if all(len(p) <= 2 for p in parts) and len(parts) == 2:
                return part  # It's an abbreviation like W.O
            else:
                return ' '.join(parts)

        # Match any sequence with one or more dot-separated words/numbers
        text = re.sub(r'\b(?:\w+\.)+\w+\b', dot_replacer, text)

    # Rule 3: Remove digits
    if "remove_digits" in rules:
        # Match digits that are NOT surrounded by letters on both sides
        # Remove floats or integers not between letters
        text = re.sub(r'(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])', '', text)

    
    #after all rules applied, remove any leading or trailing special characters and whitespace
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove trailing special characters and whitespace
    text = re.sub(r'[\s\-.,/\\]+$', '', text)

    return text


df_filter=df_updated_products[[product_id_MARA,product_desc, product_desc_update]]
#step 0: preprocess the description
df_filter.insert(df_filter.columns.get_loc(product_desc_update) + 1, rules_desc, df_filter[product_desc_update].apply(preprocess_description))

# Step 1: create tokens from the description
df_filter.insert(df_filter.columns.get_loc(rules_desc) + 1, 'tokens', df_filter[rules_desc].apply(tokenize_text))

#Step 2: create a vocabulary of tokens with their frequencies
vocab_counter = Counter(token for tokens in df_filter["tokens"] for token in tokens)

df_vocab = pd.DataFrame([
    {
        "token": token,
        "frequency": freq,    }
    for token, freq in vocab_counter.items()
])
df_vocab = df_vocab.sort_values(by="frequency", ascending=False).reset_index(drop=True)
print(f"Vocabulary created with {len(df_vocab)} unique tokens.")

#step 3: create filter vocabulary
# Define a list of stopwords to remove (you can expand this list)
custom_stopwords = {'for', 'in', 'with', 'and', 'or', 'on', 'of', 'the', 'a', 'an'}
def is_number(token):
    return bool(re.fullmatch(r'-?\d+(\.\d+)?', token))

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


#step 5: create a dictonary with elboreated tokens and all the shourtcuts

# Step 6: apply the dictionary on the dscription column



#endregion






#region step 1 - tokenization and vocabulary creation



# df_updated_products.drop(columns=['tokens'], inplace=True, errors='ignore')  # Remove existing 'lemmatized' column if it exists

# filter out to many values ...
# df_filter.insert(df_filter.columns.get_loc(product_desc_update) + 2, 'tokens_filter', df_filter.apply(
#     lambda row: [
#         token for token in row['tokens'] 
#         if token.lower() not in (row[manufacturer_model] or "").lower()
#     ],
#     axis=1
# ))
# df_check=df_filter[df_filter['tokens_filter']!= df_filter['tokens']].reset_index(drop=True)



# Step 1: Create defaultdict to collect product IDs for each token, token is the key and the value id product id
token_to_ids = defaultdict(set)
for _, row in df_filter.iterrows():
    pid = row[product_id_MARA]
    for token in row["tokens"]:
        token_to_ids[token].add(pid)




#endregion

#region Phase B: Semantic Mapping & Abbreviation Expansion
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.preprocessing import normalize

MODEL = "emilyalsentzer/Bio_ClinicalBERT"  # or "medicalai/ClinicalBERT"
MODEL = "medicalai/ClinicalBERT"  # or "medicalai/ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokens = df_vocab_filter['token'].tolist()

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


batch_size = 64
vectors = []

for i in range(0, len(tokens), batch_size):
    batch = tokens[i:i+batch_size]
    vecs = embed_tokens(batch, model, tokenizer, device)
    vectors.append(vecs)

all_vectors = np.vstack(vectors)  # <- Combine all batches
df_vocab_filter["vector"] = list(all_vectors)


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



#Mode 1: Top-k (e.g., top 5 neighbors)
vectors_norm = normalize(all_vectors, axis=1)
top_neighbors = get_faiss_neighbors(vectors_norm, df_vocab_filter['token'].tolist(), top_k=5)
df_vocab_filter['top_neighbors'] = top_neighbors

#🔹 Mode 2: Threshold-based (e.g., similarity ≥ 0.85)
top_neighbors = get_faiss_neighbors(vectors_norm, df_vocab_filter['token'].tolist(), threshold=0.9)
df_vocab_filter['th_neighbors'] = top_neighbors




def compute_neighbors(df, model_name, model_label, top_k=5, threshold=0.9, batch_size=64):
    """
    Embed tokens and compute FAISS neighbors for a given model.

    Args:
        df (pd.DataFrame): DataFrame with a 'token' column
        model_name (str): Hugging Face model name
        model_label (str): Label to differentiate model outputs in column names
        top_k (int): Number of top neighbors
        threshold (float): Similarity threshold
        batch_size (int): Embedding batch size
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Embed in batches
    tokens = df['token'].tolist()
    vectors = []

    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        vecs = embed_tokens(batch, model, tokenizer, device)
        vectors.append(vecs)

    all_vectors = np.vstack(vectors)
    vectors_norm = normalize(all_vectors, axis=1)

    # Compute neighbors
    top_neighbors = get_faiss_neighbors(vectors_norm, tokens, top_k=top_k)
    th_neighbors = get_faiss_neighbors(vectors_norm, tokens, threshold=threshold)

    # Add new columns to the DataFrame
    df[f'top_neighbors_{model_label}'] = top_neighbors
    df[f'th_neighbors_{model_label}'] = th_neighbors

    return df


MODEL1 = "emilyalsentzer/Bio_ClinicalBERT"  # or "medicalai/ClinicalBERT"
MODEL2 = "medicalai/ClinicalBERT"  # or "medicalai/ClinicalBERT"
df_vocab_filter_model1=compute_neighbors(df_vocab_filter.copy(), MODEL1, 'bio', top_k=5, threshold=0.9)
df_vocab_filter_model1_2=compute_neighbors(df_vocab_filter_model1.copy(), MODEL2, 'med', top_k=5, threshold=0.88)



#endregion

#region step1+2 combine

#🔁 Step 1: Embed full descriptions
def embed_texts(texts, model, tokenizer, device, max_length=64):
    """
    Embed a list of full text descriptions using a BERT model.
    """
    model.eval()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
    
    embeddings = outputs.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    summed = torch.sum(embeddings * attention_mask, dim=1)
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts
    return mean_pooled.cpu().numpy()


#🧰 Step 3: Full pipeline for model + description clustering
def compute_description_neighbors(df, text_col, model_name, model_label, top_k=5, threshold=0.9):
    """
    Compute semantic neighbors for product descriptions using a specified transformer model.
    This function encodes a text column using a transformer-based language model, then uses FAISS to find:
    - The top-K most similar descriptions
    - All descriptions above a similarity threshold

    These results are saved to two new columns:
    - 'top_desc_<model_label>': list of top-K closest descriptions
    - 'th_desc_<model_label>': list of descriptions with similarity above the threshold
    Args:
        df (pd.DataFrame): DataFrame with a text column to embed
        text_col (str): Name of the column containing text descriptions
        model_name (str): Hugging Face model name
        model_label (str): Label to differentiate model outputs in column names
        top_k (int): Number of top neighbors to retrieve (default is 5).
        threshold (float): Similarity threshold for neighbors (default is 0.9).
        Returns:
        pd.DataFrame: Original DataFrame with new columns for neighbors
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = df[text_col].fillna("").tolist()
    vectors = []

    # Batch process to avoid GPU overload
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = embed_texts(batch, model, tokenizer, device)
        vectors.append(vecs)

    all_vectors = np.vstack(vectors)
    vectors_norm = normalize(all_vectors, axis=1)

    # Save top-K and threshold neighbors
    top_neighbors = get_faiss_neighbors(vectors_norm, texts, top_k=top_k)
    th_neighbors = get_faiss_neighbors(vectors_norm, texts, threshold=threshold)

    df[f'top_desc_{model_label}'] = top_neighbors
    df[f'th_desc_{model_label}'] = th_neighbors
    return df


# Run for Bio_ClinicalBERT
df_filter_model1=compute_description_neighbors(
    df_filter[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]].copy(),
    text_col=product_desc_update,
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    model_label="bio"
)

df_filter_model1.insert(
    df_filter_model1.columns.get_loc('top_desc_bio') + 1,
    'top_desc_bio_tokens',
    df_filter_model1['top_desc_bio'].apply(lambda desc_list: tokenize_text(" ".join(desc_list)) if isinstance(desc_list, list) else [])
)


# Run for ClinicalBERT
df_filter_model_2=compute_description_neighbors(
    df_filter[[product_id_MARA,product_desc, product_desc_update,manufacturer_model,unit_code_number]].copy(),
    text_col=product_desc_update,
    model_name="medicalai/ClinicalBERT",
    model_label="med"
)
df_filter_model_2.insert(
    df_filter_model_2.columns.get_loc('th_desc_med') + 1,
    'th_desc_med_tokens',
    df_filter_model_2['th_desc_med'].apply(lambda desc_list: tokenize_text(" ".join(desc_list)) if isinstance(desc_list, list) else [])
)








#endregion











# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk import pos_tag
# import pandas as pd



# Download resources (if not already done)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

# lemmatizer = WordNetLemmatizer()

# # POS tag mapping function
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# # Lemmatization function for one text string
# def lemmatize_text(text):
#     tokens = word_tokenize(text)
#     tagged = pos_tag(tokens)
#     lemmatized = [
#         lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos))
#         for token, pos in tagged
#         if token.isalpha()
#     ]
#     return set(lemmatized)
# # Tokenization function (keep word order, remove punctuation)

# word = 'Patients'

# lemma = lemmatizer.lemmatize(word.lower(), pos_tag(word))

# print(f"Lemmatized '{word}' as '{lemma}'")