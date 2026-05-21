# Retail Crisis & Recovery Analysis

Python-based retail analytics project developed for a visualization and business intelligence challenge.

## Features
- Rising Star Product Detection
- 3-Day Moving Average Analysis
- Apriori Market Basket Analysis
- Comparative Retail Visualization
- Automated Excel Reporting

## Output
- retail_insight.xlsx
- rising_star_index.png
- rising_star_actual.png

## Final Score
84.45 / 100

### Score Breakdown
| Component | Score |
|---|---|
| Excel Scoring | 54.45 / 70 |
| Visualization | 15 / 15 |
| Actual Visualization | 15 / 15 |

## Known Limitations
Some edge cases in the generated Excel result still differ from the scorer expectation.

Expected rows: 18  
Correct rows: 14

Possible causes:
- association rule ordering
- edge-case filtering
- deterministic scorer formatting

## Tech Stack
- Python
- Pandas
- Matplotlib
- Mlxtend
- OpenPyXL
