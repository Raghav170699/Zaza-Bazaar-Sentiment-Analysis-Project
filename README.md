# Social Media Analytics for Restaurant Performance Analysis

## Project Overview
Analysis of customer sentiment and identifying areas of improvement for Zaza Bazaar (a restaurant in Bristol, UK) using social media analytics and machine learning techniques.

## Data Source
- Platform: TripAdvisor
- Sample Size: 5,334 customer reviews
- Data Collection Method: Web scraping using Octoparse

## Methodology
### Data Processing
- Data cleaning and preprocessing
- Removal of duplicate entries
- Text tokenization and stemming
- Stop words removal

### Analysis Techniques
1. **Sentiment Analysis**
   - Used VADER (Valence Aware Dictionary and Sentiment Reasoner)
   - Categorized reviews into Positive, Negative, and Neutral
   - NLTK (Natural Language Toolkit) for text processing

2. **Visualization**
   - Word Cloud generation for negative reviews
   - Bar charts for sentiment distribution

3. **Machine Learning Models**
   - K-means Clustering (5 clusters)
   - LDA (Latent Dirichlet Allocation) Topic Modeling

## Key Findings
- Majority of customers expressed positive sentiment
- ~1000 customers reported negative experiences
- Main pain points identified:
  - Service quality
  - Wait times during busy hours
  - Food quality concerns

## Recommendations
1. **Customer Retention**
   - Focus on maintaining food quality standards
   - Expand menu offerings
   - Implement loyalty programs

2. **Family Experience**
   - Create dedicated kid-friendly sections
   - Consider adding live music
   - Optimize pricing for family budgets

3. **Sustainability**
   - Minimize waste
   - Optimize energy usage
   - Reduce carbon footprint

4. **Service Improvement**
   - Enhanced staff training
   - Focus on hospitality experience
   - Improved customer interaction

## Limitations
- Limited dataset size
- Sentiment analysis accuracy constraints
- Pre-set metric requirements for clustering models

## Technologies Used
- Python
- NLTK
- Pandas
- VADER
- Octoparse (for web scraping)

## Future Research Directions
- Expand data sources (Yelp, Twitter, Facebook)
- Improve sentiment analysis accuracy
- Optimize clustering parameters
- Enhanced natural language processing capabilities


