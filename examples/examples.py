# {
#   "lexical_tokens": ["is", "delq", "90d"],
#   "expanded_tokens": ["Indicator", "Delinquency", "90 Days"],
#   "table_context": "credit_account_snapshot",
#   "domain": "Retail Banking",
#   "data_type": "BOOLEAN",
#   "derivation_rule": "days_past_due >= 90",
#   "approved_glossary_terms": [
#     "Delinquency",
#     "Days Past Due"
#   ],
#   "naming_rules": {
#     "boolean_prefix": "Indicator",
#     "max_words": 6
#   }
# }


# {
#   "metadata_id": "string",

#   "domain": "string",
#   "business_context": "string",
#   "system": "string",
#   "schema": "string",
#   "table": "string",
#   "column": "string",

#   "data_type": "STRING | INTEGER | DECIMAL | BOOLEAN | DATE | TIMESTAMP",
#   "nullable": true,
#   "unit": "string | null",

#   "lexical_tokens": ["string"],
#   "expanded_tokens": ["string"],

#   "derivation_logic": "string | null",
#   "value_encoding": "string | null",

#   "sample_value_profile": {
#     "min": "number | null",
#     "max": "number | null",
#     "distinct_count": "number | null",
#     "true_ratio": "number | null"
#   },

#   "approved_glossary_terms": ["string"],

#   "prior_approved_examples": [
#     {
#       "column": "string",
#       "business_name": "string"
#     }
#   ],

#   "naming_constraints": {
#     "max_words": 6,
#     "boolean_suffix": "Indicator",
#     "numeric_suffix": "Amount | Count | Rate | Score",
#     "date_suffix": "Date",
#     "allow_parenthetical": true | false
#   }
# }




# #output example schema

# {
#   "predicted_business_name": "string",

#   "approval_status": "REVIEW_REQUIRED",

#   "confidence_score": 0.0,
#   "ambiguity_score": 0.0,

#   "risk_flags": [
#     "NO_DERIVATION_LOGIC",
#     "UNKNOWN_UNIT",
#     "ABBREVIATION_UNMAPPED",
#     "VALUE_ENCODING_UNKNOWN",
#     "DOMAIN_CONFLICT"
#   ],

#   "assumptions": ["string"],

#   "justification": "string"
# }
