"""
AFE OUTPUT NORMALIZATION

Role:
- Standardize outputs across all executed models
- Ensure schema consistency for tables, charts, and comparisons
- Attach execution metadata and confidence framing

Must:
- Preserve model disagreement
- Preserve uncertainty
- Emit audit-ready outputs

Must NOT:
- Rank models
- Highlight winners or losers
- Suppress poor performance
"""
