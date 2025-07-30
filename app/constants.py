
# app/constants.py
print("✅ constants.py is running")
# Constants used across the application

#keys
new_openAI_key='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'


#sarel catalog constants
product_id_CE1SARL='ARTNR'
product_id_MARA='MATNR'
product_desc='full_desc'
product_code_sub_field='MATKL'
product_desc_sub_field='WGBEZ'
product_dv_id='SPART'
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


#constants for filtering dvs (spart)
ls_dvs=['11','12','22','25']

#new constants for used in the application
product_order_basic_unit='BSTME_merge_MEINS'

#new constants for used in the application - for product desc update or thershold
product_desc_update='full_desc_update'
rules_desc='rules_desc'
final_desc='desc_fix'
th_manu= 85
th_desc = 85


#constants for input data from suppliers
input_supplier_name='Supplier Name'
input_provider_num='Provider number supplier'
input_id_number='Id number'
input_desc='Description'

#constants for db
#tables name
products_desc_table='products_new_desc'
shortcut_table='shortcut_dict'
prompt_table='prompt_result'


#columns
inserted_date='inserted_at'
updated_date='updated_at'
prompt_date='prompt_date'




