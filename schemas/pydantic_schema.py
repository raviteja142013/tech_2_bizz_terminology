
from typing import List, Optional, Any
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


# --- Enumerations / Literals ---

# Matches: "STRING | INTEGER | DECIMAL | BOOLEAN | DATE | TIMESTAMP"
DataType = Literal["STRING", "INTEGER", "DECIMAL", "BOOLEAN", "DATE", "TIMESTAMP"]

# Matches: "Indicator"
BooleanSuffix = Literal["Indicator"]

# Matches: "Amount | Count | Rate | Score"
NumericSuffix = Literal["Amount", "Count", "Rate", "Score"]

# Matches: "Date"
DateSuffix = Literal["Date"]


# --- Nested Models ---

class SampleValueProfile(BaseModel):
    """
    Represents optional profiling stats for a column.
    - min/max: could be numeric; allow None
    - distinct_count: integer; allow None
    - true_ratio: for booleans; allow None
    """
    min: Optional[Any] = Field(default=None)
    max: Optional[Any] = Field(default=None)
    distinct_count: Optional[int] = Field(default=None, ge=0)
    true_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PriorApprovedExample(BaseModel):
    """
    A prior approved mapping from a technical column to a business-friendly name.
    """
    column: str
    business_name: str


class NamingConstraints(BaseModel):
    """
    Rules governing how business names should be formed.
    """
    max_words: int = Field(..., ge=1, description="Maximum number of words allowed in the business name.")
    boolean_suffix: Optional[BooleanSuffix]
    numeric_suffix: Optional[NumericSuffix]
    date_suffix: Optional[DateSuffix]
    allow_parenthetical: bool


# --- Main Model ---

class ColumnMetadata(BaseModel):
    """
    Pydantic model for the provided JSON structure describing column business naming context.
    """
    model_config = ConfigDict(populate_by_name=True)

    # Core identifiers / context
    metadata_id: str
    domain: str
    business_context: str
    system: str
    attribute_schema: str
    table: str
    column: str

    # Technical characteristics
    data_type: DataType
    nullable: bool
    unit: Optional[str] = None  # e.g., "USD", "kg", may be None

    # Tokens
    lexical_tokens: List[str] = Field(default_factory=list)
    expanded_tokens: Optional[List[str]]= Field(default_factory=list)

    # Additional semantics
    derivation_logic: Optional[str] = None
    value_encoding: Optional[str] = None

    # Profiling
    sample_value_profile: Optional[SampleValueProfile] = None

    # Governance info
    approved_glossary_terms: Optional[List[str]]=  Field(default_factory=list)

    # Prior examples
    prior_approved_examples: Optional[List[PriorApprovedExample]] =   Field(default_factory=list)

    # Naming constraints/rules
    naming_constraints: Optional[NamingConstraints]


# --- Enumerations / Literals ---
ApprovalStatus = Literal["REVIEW_REQUIRED", "APPROVED", "REJECTED"]

RiskFlag = Literal[
    "NO_DERIVATION_LOGIC",
    "UNKNOWN_UNIT",
    "ABBREVIATION_UNMAPPED",
    "VALUE_ENCODING_UNKNOWN",
    "DOMAIN_CONFLICT"
]


# --- Main Model ---
class BusinessNamePrediction(BaseModel):
    """
    Represents the predicted business name and its evaluation details.
    """

    predicted_business_name: str

    approval_status: ApprovalStatus

    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    ambiguity_score: float = Field(..., ge=0.0, le=1.0, description="Ambiguity score between 0 and 1.")

    risk_flags: Optional[List[RiskFlag]] =  Field(default_factory=list)

    assumptions: Optional[List[str]] = Field(default_factory=list)

    justification: str
