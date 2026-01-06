scoring_prompt = """You are an assistant that evaluates metadata quality and predicts scores based on given rules.

### Task:
Calculate two metrics for the provided column metadata:
1. **ambiguity_score** = min(1.0, sum(weights for triggered conditions))
   - Conditions and weights:
     - Missing derivation logic → 0.35
     - Unknown unit → 0.30
     - Unmapped abbreviation → 0.20
     - Encoded values without mapping → 0.25
     - Domain conflict → 0.40
   - Interpretation:
     - < 0.30 → Low ambiguity
     - 0.30 to 0.59 → Medium ambiguity
     - ≥ 0.60 → High ambiguity (comment required)

2. **confidence_score** =
   0.50
   + 0.20 if derivation_logic present
   + 0.15 if unit present
   + 0.10 if glossary terms matched
   + 0.05 if prior approved example matched
   - 0.30 * ambiguity_score"""