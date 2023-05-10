# Define the training and validation datasets
# train_dataset = [("The customer's name is John Smith.", "customer_name"),
#     ("The order total is $50.", "order_total"),
#     ("The product SKU is 12345.", "product_SKU"),
#     ("The shipping address is 123 Main St.", "shipping_address")]
# # list of tuples (input, label)
# val_dataset = [("The customer's name is Anal Sarkar.", "customer_name"),
#     ("The order total is $1000.", "order_total"),
#     ("The product SKU is 22345.", "product_SKU"),
#     ("The shipping address is 456 Main St.", "shipping_address")]

# Define the label mapping
#label_map = {"customer_name": 0, "order_total": 1, "product_SKU": 2, "shipping_address": 3}
# Add more labels as needed


# Define the label mapping
label_map = {"COMPANY_CODE": 0, "INVOICE": 1, "PURCHASE": 2, "INVOICE_PURCHASE": 3}
# Add more labels as needed

train_dataset = [("gl_account company_code  customer_name debit  credit amount", "COMPANY_CODE"),
    ("invoice_id Item_name Item_price amount", "INVOICE"),
    ("purchase_order_id vendor_name price_amount", "PURCHASE"),
    ("invoice_id purchase_order_id", "INVOICE_PURCHASE")]
# list of tuples (input, label)
val_dataset = [("gl_account company_code  customer_name debit  credit amount", "COMPANY_CODE"),
    ("invoice_id Item_name Item_price amount", "INVOICE"),
    ("purchase_order_id vendor_name price_amount", "PURCHASE"),
    ("invoice_id purchase_order_id", "INVOICE_PURCHASE")]