# example 1

{
  "metadata_id": "MD-001",
  "domain": "Retail Banking",
  "business_context": "Credit Risk Monitoring",
  "system": "CoreBank",
  "schema": "risk",
  "table": "credit_account_snapshot",
  "column": "is_delq_90d",
  "data_type": "BOOLEAN",
  "nullable": false,
  "unit": null,
  "lexical_tokens": ["is", "delq", "90d"],
  "expanded_tokens": ["Indicator", "Delinquency", "90 Days"],
  "derivation_logic": "days_past_due >= 90",
  "value_encoding": null,
  "sample_value_profile": {
    "min": null,
    "max": null,
    "distinct_count": 2,
    "true_ratio": 0.14
  },
  "approved_glossary_terms": ["Delinquency", "Days Past Due"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 6,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": false
  }
}


{
  "predicted_business_name": "90-Day Delinquency Indicator",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.92,
  "ambiguity_score": 0.08,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Boolean indicator derived from days past due threshold of 90 days"
}



# example 2

{
  "metadata_id": "MD-002",
  "domain": "Payments",
  "business_context": "Transaction Analytics",
  "system": "PaymentsHub",
  "schema": "fact",
  "table": "daily_transaction_summary",
  "column": "total_txn_amt_usd",
  "data_type": "DECIMAL",
  "nullable": false,
  "unit": "USD",
  "lexical_tokens": ["total", "txn", "amt", "usd"],
  "expanded_tokens": ["Total", "Transaction", "Amount", "USD"],
  "derivation_logic": "sum(transaction_amount_usd) per day",
  "value_encoding": null,
  "sample_value_profile": {
    "min": 0,
    "max": 12500000,
    "distinct_count": null,
    "true_ratio": null
  },
  "approved_glossary_terms": ["Transaction", "Amount"],
  "prior_approved_examples": [
    {
      "column": "daily_txn_amt_usd",
      "business_name": "Daily Transaction Amount (USD)"
    }
  ],
  "naming_constraints": {
    "max_words": 6,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": true
  }
}


{
  "predicted_business_name": "Total Transaction Amount (USD)",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.88,
  "ambiguity_score": 0.12,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Summed monetary transaction amount in USD at daily granularity"
}


# example 3

{
  "metadata_id": "MD-003",
  "domain": "Customer",
  "business_context": "Customer Profile",
  "system": "CRM",
  "schema": "dim",
  "table": "customer_profile",
  "column": "acct_open_dt",
  "data_type": "DATE",
  "nullable": false,
  "unit": null,
  "lexical_tokens": ["acct", "open", "dt"],
  "expanded_tokens": ["Account", "Open", "Date"],
  "derivation_logic": "date when account was opened in source system",
  "value_encoding": null,
  "sample_value_profile": {
    "min": "1998-01-01",
    "max": "2025-01-01",
    "distinct_count": null,
    "true_ratio": null
  },
  "approved_glossary_terms": ["Account"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 5,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": false
  }
}

{
  "predicted_business_name": "Account Open Date",
  "approval_status": "REVIEW_REQUIRED",
  "confidence_score": 0.85,
  "ambiguity_score": 0.10,
  "risk_flags": [],
  "assumptions": [],
  "justification": "Date representing when the account was opened"
}
