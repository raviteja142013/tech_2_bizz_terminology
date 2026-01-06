
bad_examples = {
    1 : {
  "metadata_id": "MD-010",
  "domain": "Customer",
  "business_context": "Customer Status",
  "system": "CRM",
  "attribute_schema": "dim",
  "table": "customer_profile",
  "column": "cust_stat_cd",
  "data_type": "STRING",
  "nullable": False,
  "unit": None,
  "lexical_tokens": ["cust", "stat", "cd"],
  "expanded_tokens": ["Customer", "Status", "Code"],
  "derivation_logic": None,
  "value_encoding": "A/I/S",
  "sample_value_profile": {
    "min": None,
    "max": None,
    "distinct_count": 3,
    "true_ratio": None
  },
  "approved_glossary_terms": [],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 5,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
},

    2 : {
  "metadata_id": "MD-011",
  "domain": "Accounts",
  "business_context": "Account Flags",
  "system": "LegacyCore",
  "attribute_schema": "legacy",
  "table": "acct_flags",
  "column": "is_active",
  "data_type": "BOOLEAN",
  "nullable": True,
  "unit": None,
  "lexical_tokens": ["is", "active"],
  "expanded_tokens": ["Indicator", "Active"],
  "derivation_logic": None,
  "value_encoding": "Y/N",
  "sample_value_profile": {
    "min": None,
    "max": None,
    "distinct_count": 2,
    "true_ratio": 0.82
  },
  "approved_glossary_terms": [],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 4,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
}
,

    3 : {
  "metadata_id": "MD-012",
  "domain": "Marketing",
  "business_context": "Campaign Metrics",
  "system": "MarTech",
  "attribute_schema": "fact",
  "table": "campaign_performance",
  "column": "score_val",
  "data_type": "DECIMAL",
  "nullable": False,
  "unit": None,
  "lexical_tokens": ["score", "val"],
  "expanded_tokens": ["Score", "Value"],
  "derivation_logic": None,
  "value_encoding": None,
  "sample_value_profile": {
    "min": 0.12,
    "max": 0.97,
    "distinct_count": None,
    "true_ratio": None
  },
  "approved_glossary_terms": [],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 4,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Score",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
}


}



bad_outputs = {
    1 : {
  "predicted_business_name": "Unspecified Customer Status Code",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.29,
  "ambiguity_score": 0.74,
  "risk_flags": ["VALUE_ENCODING_UNKNOWN", "NO_DERIVATION_LOGIC"],
  "assumptions": [
    "Code represents a customer status classification",
    "Meaning of encoded values is undocumented"
  ],
  "justification": "Status code lacks value mapping and derivation context"
},

    2 : {
  "predicted_business_name": "Unspecified Active Indicator",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.33,
  "ambiguity_score": 0.61,
  "risk_flags": ["NO_DERIVATION_LOGIC", "VALUE_ENCODING_UNKNOWN"],
  "assumptions": [
    "Indicator reflects an active or inactive state",
    "Criteria for active status is undocumented"
  ],
  "justification": "Boolean indicator lacks definition and activation criteria"
},

    3 : {
  "predicted_business_name": "Unspecified Score Value",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.27,
  "ambiguity_score": 0.78,
  "risk_flags": ["NO_DERIVATION_LOGIC", "UNKNOWN_UNIT"],
  "assumptions": [
    "Score represents a normalized numeric metric",
    "Scoring methodology is undocumented"
  ],
  "justification": "Metric lacks definition, unit, and calculation logic"
}


 
}


# #example 1

# {
#   "metadata_id": "MD-004",
#   "domain": "Payments",
#   "business_context": "Transaction Processing",
#   "system": "LegacyPay",
#   "schema": "raw",
#   "table": "txn_log",
#   "column": "txn_amt",
#   "data_type": "DECIMAL",
#   "nullable": true,
#   "unit": null,
#   "lexical_tokens": ["txn", "amt"],
#   "expanded_tokens": ["Transaction", "Amount"],
#   "derivation_logic": null,
#   "value_encoding": null,
#   "sample_value_profile": {
#     "min": -5000,
#     "max": 5000,
#     "distinct_count": null,
#     "true_ratio": null
#   },
#   "approved_glossary_terms": [],
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
#   "predicted_business_name": "Unspecified Transaction Amount",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.38,
#   "ambiguity_score": 0.65,
#   "risk_flags": ["UNKNOWN_UNIT", "NO_DERIVATION_LOGIC"],
#   "assumptions": [
#     "Value represents a monetary amount",
#     "Currency is not specified"
#   ],
#   "justification": "Numeric transaction value lacks unit, currency, and derivation context"
# }



# # example 2

# {
#   "metadata_id": "MD-005",
#   "domain": "Customer",
#   "business_context": "Account Flags",
#   "system": "Mainframe",
#   "schema": "legacy",
#   "table": "acct_flags",
#   "column": "flg_x9",
#   "data_type": "BOOLEAN",
#   "nullable": true,
#   "unit": null,
#   "lexical_tokens": ["flg", "x9"],
#   "expanded_tokens": ["Flag"],
#   "derivation_logic": null,
#   "value_encoding": "0/1",
#   "sample_value_profile": {
#     "min": null,
#     "max": null,
#     "distinct_count": 2,
#     "true_ratio": 0.47
#   },
#   "approved_glossary_terms": [],
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
#   "predicted_business_name": "Unspecified Account Flag Indicator",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.22,
#   "ambiguity_score": 0.82,
#   "risk_flags": ["ABBREVIATION_UNMAPPED", "NO_DERIVATION_LOGIC", "VALUE_ENCODING_UNKNOWN"],
#   "assumptions": [
#     "Flag represents a binary condition",
#     "Business meaning of flag is undocumented"
#   ],
#   "justification": "Boolean flag lacks derivation logic and has unmapped identifier"
# }



# # example 3

# {
#   "metadata_id": "MD-006",
#   "domain": "Lending",
#   "business_context": "Loan Metrics",
#   "system": "LoanSys",
#   "schema": "metrics",
#   "table": "loan_summary",
#   "column": "rate_val",
#   "data_type": "DECIMAL",
#   "nullable": false,
#   "unit": null,
#   "lexical_tokens": ["rate", "val"],
#   "expanded_tokens": ["Rate", "Value"],
#   "derivation_logic": null,
#   "value_encoding": null,
#   "sample_value_profile": {
#     "min": 0.01,
#     "max": 0.35,
#     "distinct_count": null,
#     "true_ratio": null
#   },
#   "approved_glossary_terms": [],
#   "prior_approved_examples": [],
#   "naming_constraints": {
#     "max_words": 6,
#     "boolean_suffix": "Indicator",
#     "numeric_suffix": "Rate",
#     "date_suffix": "Date",
#     "allow_parenthetical": false
#   }
# }


# {
#   "predicted_business_name": "Unspecified Rate Value",
#   "approval_status": "REVIEW_REQUIRED",
#   "confidence_score": 0.31,
#   "ambiguity_score": 0.71,
#   "risk_flags": ["UNKNOWN_UNIT", "NO_DERIVATION_LOGIC"],
#   "assumptions": [
#     "Rate represents a proportional value",
#     "Specific rate type is not documented"
#   ],
#   "justification": "Rate column lacks definition, unit, and calculation logic"
# }
