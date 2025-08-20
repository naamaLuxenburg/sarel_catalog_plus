import os
# from sentence_transformers import SentenceTransformer


# app/constants.py
print("✅ constants.py is running")
# Constants used across the application

#workspace root path
workspace_root = os.environ.get("PYTHONPATH", os.getcwd())

#model embeddings
model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
col_embedding_index='embedding_index'
# model = SentenceTransformer(model_name)


#sarel catalog constants
product_id_CE1SARL='ARTNR'
product_id_MARA='MATNR'
product_desc='full_desc'
product_code_sub_field='MATKL'
product_desc_sub_field='WGBEZ'
product_dv_id='SPART' #full word.
product_dv_desc='VTEXT'
product_category='MTPOS_MARA'
product_basic_unit='MEINS'
product_order_unit='BSTME'
manufacturer_model='MFRPN'
supplier_name='NAME1'

unit_code='Code1'
unit_desc='MSEHL_x'
unit_code_number='Code_number'

#med data
med_indication='Indication'
med_generic='fix_Generic_Name'
med_concentration='Concentration_Fix'
med_dosage='Dosage_Form_org'
med_dose='dose'


#price field
price_client='HIENR'
price_in_coin='KBETR'
price_coin='KONWA'
price_unit='KMEIN'
price_unit_ILS='KMEIN_ILS'



#constants for filtering dvs (spart)
ls_dvs=['11','12','22','25']

#new constants for used in the application
product_order_basic_unit='BSTME_merge_MEINS'

#new constants for used in the application - for product desc update or thershold
product_desc_update='full_desc_update'
rules_desc='rules_desc'
final_desc='desc_fix'
# th_manu= 85
# th_desc = 85


#constants for input data from suppliers - need to change
# input_supplier_name='Supplier Name'
# input_provider_num='Provider number supplier'
# input_desc='Description' -> input from the user: col_input_desc

#constants for input data- application addition
col_input_desc_norm='Description_norm'  ###org to change input_desc_norm
col_input_id='Serial_id_number' ###input_id

#for the results AI
col_input_result_tokens='input_tokens'
col_input_result_match_desc='matched_description'
col_input_result_match_tokens='matched_tokens'
col_input_result_product='db_MATNR'
col_input_result_product_dv='db_SPART'
col_input_result_generic='db_generic'
col_input_result_faiss_score='faiss_score'
col_input_result_faiss_rank='faiss_rank'
col_input_result_token_score='token_score'
col_input_result_generic_score='generic_score'
col_input_result_comb_faiss_token='combination_faiss_token_score'
col_input_result_comb_faiss_generic='combination_faiss_generic_score'
col_input_result_final_score='final_score'
col_input_result_num='Result_number'



#constants for db catalog new
#tables name
products_desc_table='products_new_desc'
shortcut_table='shortcut_dict'
prompt_table='prompt_result'

#columns for shortcut_table
col_shortcut='shortcut'
col_meaning='meaning'
col_source='source'
col_accuracy='accuracy'


#dates columns
inserted_date='inserted_at'
updated_date='updated_at'
prompt_date='prompt_date'




