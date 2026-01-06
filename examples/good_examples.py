

good_examples = {
    1 : {
  "metadata_id": "MD-007",
  "domain": "Customer",
  "business_context": "Customer Profile",
  "system": "CRM",
  "attribute_schema": "dim",
  "table": "customer_profile",
  "column": "cust_id",
  "data_type": "STRING",
  "nullable": False,
  "unit": None,
  "lexical_tokens": ["cust", "id"],
  "expanded_tokens": ["Customer", "Identifier"],
  "derivation_logic": "unique identifier assigned to each customer",
  "value_encoding": None,
  "sample_value_profile": {
    "min": None,
    "max": None,
    "distinct_count": 1250000,
    "true_ratio": None
  },
  "approved_glossary_terms": ["Customer"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 4,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
},
    2 : {
  "metadata_id": "MD-008",
  "domain": "Accounts",
  "business_context": "Account Analytics",
  "system": "CoreBank",
  "attribute_schema": "fact",
  "table": "account_activity_summary",
  "column": "txn_cnt_30d",
  "data_type": "INTEGER",
  "nullable": False,
  "unit": "COUNT",
  "lexical_tokens": ["txn", "cnt", "30d"],
  "expanded_tokens": ["Transaction", "Count", "30 Days"],
  "derivation_logic": "count of transactions in last 30 days",
  "value_encoding": None,
  "sample_value_profile": {
    "min": 0,
    "max": 842,
    "distinct_count": None,
    "true_ratio": None
  },
  "approved_glossary_terms": ["Transaction"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 6,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Count",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
},
    3 : {
  "metadata_id": "MD-009",
  "domain": "Accounts",
  "business_context": "Balance Management",
  "system": "CoreBank",
  "attribute_schema": "fact",
  "table": "account_daily_balance",
  "column": "end_bal_usd",
  "data_type": "DECIMAL",
  "nullable": False,
  "unit": "USD",
  "lexical_tokens": ["end", "bal", "usd"],
  "expanded_tokens": ["Ending", "Balance", "USD"],
  "derivation_logic": "account balance at end of business day",
  "value_encoding": None,
  "sample_value_profile": {
    "min": -1200.50,
    "max": 2500000.00,
    "distinct_count": None,
    "true_ratio": None
  },
  "approved_glossary_terms": ["Balance"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 5,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": True
  }
}



}


good_outputs = {
    1 : {
  "predicted_business_name": "Customer Identifier",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.91,
  "ambiguity_score": 0.05,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Unique identifier used to reference a customer record"
},
    2 : {
  "predicted_business_name": "30-Day Transaction Count",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.87,
  "ambiguity_score": 0.10,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Count of transactions occurring within the last 30 days"
},
    3 : {
  "predicted_business_name": "Ending Account Balance (USD)",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.89,
  "ambiguity_score": 0.12,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Monetary account balance captured at end of business day"
}



}




# # example 1

# {
#   "metadata_id": "MD-001",
#   "domain": "Retail Banking",
#   "business_context": "Credit Risk Monitoring",
#   "system": "CoreBank",
#   "schema": "risk",
#   "table": "credit_account_snapshot",
#   "column": "is_delq_90d",
#   "data_type": "BOOLEAN",
#   "nullable": false,
#   "unit": null,
#   "lexical_tokens": ["is", "delq", "90d"],
#   "expanded_tokens": ["Indicator", "Delinquency", "90 Days"],
#   "derivation_logic": "days_past_due >= 90",
#   "value_encoding": null,
#   "sample_value_profile": {
#     "min": null,
#     "max": null,
#     "distinct_count": 2,
#     "true_ratio": 0.14
#   },
#   "approved_glossary_terms": ["Delinquency", "Days Past Due"],
#   "prior_approved_examples": [],
#   "naming_constraints": {
#     "max_words": 6,
#     "boolean_suffix": "Indicator",
#     "numeric_suffix": "Amount",
#     "date_suffix": "Date",
#     "allow_parenthetical": false
#   }
# }


# {
#   "predicted_business_name": "90-Day Delinquency Indicator",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.92,
#   "ambiguity_score": 0.08,
#   "risk_flags": [],
#   "assumptions": [],
#   "justification": "Boolean indicator derived from days past due threshold of 90 days"
# }



# # example 2

# {
#   "metadata_id": "MD-002",
#   "domain": "Payments",
#   "business_context": "Transaction Analytics",
#   "system": "PaymentsHub",
#   "schema": "fact",
#   "table": "daily_transaction_summary",
#   "column": "total_txn_amt_usd",
#   "data_type": "DECIMAL",
#   "nullable": false,
#   "unit": "USD",
#   "lexical_tokens": ["total", "txn", "amt", "usd"],
#   "expanded_tokens": ["Total", "Transaction", "Amount", "USD"],
#   "derivation_logic": "sum(transaction_amount_usd) per day",
#   "value_encoding": null,
#   "sample_value_profile": {
#     "min": 0,
#     "max": 12500000,
#     "distinct_count": null,
#     "true_ratio": null
#   },
#   "approved_glossary_terms": ["Transaction", "Amount"],
#   "prior_approved_examples": [
#     {
#       "column": "daily_txn_amt_usd",
#       "business_name": "Daily Transaction Amount (USD)"
#     }
#   ],
#   "naming_constraints": {
#     "max_words": 6,
#     "boolean_suffix": "Indicator",
#     "numeric_suffix": "Amount",
#     "date_suffix": "Date",
#     "allow_parenthetical": true
#   }
# }


# {
#   "predicted_business_name": "Total Transaction Amount (USD)",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.88,
#   "ambiguity_score": 0.12,
#   "risk_flags": [],
#   "assumptions": [],
#   "justification": "Summed monetary transaction amount in USD at daily granularity"
# }


# # example 3

# {
#   "metadata_id": "MD-003",
#   "domain": "Customer",
#   "business_context": "Customer Profile",
#   "system": "CRM",
#   "schema": "dim",
#   "table": "customer_profile",
#   "column": "acct_open_dt",
#   "data_type": "DATE",
#   "nullable": false,
#   "unit": null,
#   "lexical_tokens": ["acct", "open", "dt"],
#   "expanded_tokens": ["Account", "Open", "Date"],
#   "derivation_logic": "date when account was opened in source system",
#   "value_encoding": null,
#   "sample_value_profile": {
#     "min": "1998-01-01",
#     "max": "2025-01-01",
#     "distinct_count": null,
#     "true_ratio": null
#   },
#   "approved_glossary_terms": ["Account"],
#   "prior_approved_examples": [],
#   "naming_constraints": {
#     "max_words": 5,
#     "boolean_suffix": "Indicator",
#     "numeric_suffix": "Amount",
#     "date_suffix": "Date",
#     "allow_parenthetical": false
#   }
# }

# {
#   "predicted_business_name": "Account Open Date",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.85,
#   "ambiguity_score": 0.10,
#   "risk_flags": [],
#   "assumptions": [],
#   "justification": "Date representing when the account was opened"
# }
