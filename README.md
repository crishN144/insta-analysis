# Instagram Reach Insight: Analyzing and Predicting Post Performance

## Project Overview

This project provides a comprehensive analysis of Instagram post reach to help content creators adapt to changes in Instagram’s algorithm. Utilizing Python and various data science libraries, the analysis offers insights into optimizing Instagram strategies.

### What Was Done

- **Data Cleaning and Preprocessing**: Addressed missing values, standardized formats, and prepared the dataset for detailed examination.
- **Exploratory Data Analysis (EDA)**: Examined patterns in impressions and engagement metrics.
- **Content Analysis**: Used word clouds to explore common themes and hashtags in post captions.
- **Engagement and Reach Analysis**: Investigated correlations between engagement metrics (likes, comments, shares, saves) and post reach.
- **Conversion Rate Calculation**: Calculated the rate of conversion from profile visits to new followers.
- **Reach Prediction Model**: Developed a Passive Aggressive Regressor model to predict post reach with 71% accuracy.

### Key Findings

1. **Impression Distribution**:
   - 44% from home feeds
   - 34% from hashtags
   - 19% from the explore page
   - Hashtags are effective for reaching new audiences; explore page contributes less.

2. **Content Analysis**:
   - Word clouds identify frequent words and hashtags, revealing popular themes.

3. **Engagement and Reach**:
   - Likes strongly correlate with impressions.
   - Comments have minimal effect on reach.
   - Shares and saves moderately boost impressions.
   - Encouraging likes and saves can enhance post visibility.

4. **Conversion Rate**:
   - 41% conversion rate from profile visits to new followers.
   - Profile visits are linearly related to follower growth.

5. **Reach Prediction**:
   - The Passive Aggressive Regressor model predicts post reach with 71% accuracy.
   - Influencers can leverage this model to estimate post performance.

## Skills Demonstrated

- Data Cleaning and Preprocessing
- Exploratory Data Analysis and Visualization
- Statistical Analysis and Correlation
- Natural Language Processing (Word Clouds)
- Machine Learning (Passive Aggressive Regressor)
- Python Programming (pandas, numpy, matplotlib, seaborn, scikit-learn)

## Dataset

### Description

The dataset includes Instagram post metrics such as impressions, engagement metrics, and post content, manually collected from Instagram Insights.

### Key Attributes

| Attribute      | Description                   | Data Type | Example        |
|----------------|-------------------------------|-----------|----------------|
| Impressions    | Total impressions on the post | Integer   | 3920           |
| From Home      | Impressions from home feed    | Integer   | 2586           |
| From Hashtags  | Impressions from hashtags     | Integer   | 1028           |
| From Explore   | Impressions from explore page | Integer   | 619            |
| Likes          | Number of likes on the post   | Integer   | 162            |
| Comments       | Number of comments on the post| Integer   | 9              |
| Shares         | Number of times post was shared| Integer   | 5              |
| Saves          | Number of times post was saved| Integer   | 56             |
| Profile Visits | Visits to profile from the post| Integer   | 35             |
| Follows        | New followers from the post   | Integer   | 2              |
| Caption        | Caption text of the post      | String    | "Example caption" |
| Hashtags       | Hashtags used in the post     | String    | "#example #tags"  |


## Visualizations

1. **Donut Chart Showing the Distribution of Impressions from Different Sources**

```python
import matplotlib.pyplot as plt

# Summarize data
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

# Prepare data for plotting
labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

# Create donut chart
plt.pie(values, labels=labels, autopct='%.0f%%', startangle=140)
plt.title("Distribution of Impressions from Different Sources")
plt.show()
```

<div align="center">
    <img width="431" alt="Donut Chart Showing the Distribution of Impressions from Different Sources" src="https://github.com/user-attachments/assets/ae650cb0-74d0-4d2b-91d1-9fa9cf3f6b48">
    <p><strong>Distribution of Impressions from Different Sources</strong></p>
</div>

- The donut chart illustrates that 44% of impressions come from home feeds, 34% from hashtags, 19% from the explore page, and 3% from other sources. This emphasizes the significant role of hashtags in reaching new audiences, while the explore page contributes less.

2. **Word Clouds Displaying the Most Common Words and Hashtags in Captions**

```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Combine all captions into a single string
text = " ".join(caption for caption in data.Caption)

# Generate word cloud
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display word cloud
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Post Captions")
plt.show()
```

<div align="center">
    <img width="610" alt="Word Cloud Displaying the Most Common Words and Hashtags in Captions" src="https://github.com/user-attachments/assets/63bd6b49-9abc-4944-933e-2dd757346397">
    <p><strong>Word Cloud of Post Captions</strong></p>
</div>

- The word cloud highlights the most frequently used words and hashtags in post captions, offering insights into the topics and themes that resonate with the audience and informing content strategy.

3. **Scatter Plot with Regression Line Illustrating the Relationship Between Engagement Metrics and Post Impressions**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create scatter plot with regression line
sns.lmplot(data=data, x="Impressions", y="Likes", scatter_kws={"s": 20}, line_kws={"lw":1})
plt.title("Relationship Between Likes and Impressions")
plt.xlim(0, 15000)
plt.ylim(0, 500)
plt.show()
```

<div align="center">
    <img width="444" alt="Scatter Plot with Regression Line Illustrating the Relationship Between Engagement Metrics and Post Impressions" src="https://github.com/user-attachments/assets/fd505f0a-3b82-46de-8e14-945dbfd5a9ad">
    <p><strong>Relationship Between Likes and Impressions</strong></p>
</div>

- This scatter plot with a regression line reveals a strong positive correlation between likes and post impressions, indicating that encouraging more likes can enhance post reach.

4. **Scatter Plot Showing the Linear Relationship Between Profile Visits and New Followers**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create scatter plot with regression line
sns.lmplot(data=data, x="Profile Visits", y="Follows", scatter_kws={"s": 30}, line_kws={"lw":1})
plt.title("Relationship Between Profile Visits and Followers Gained")
plt.show()
```

<div align="center">
    <img width="529" alt="Scatter Plot Showing the Linear Relationship Between Profile Visits and New Followers" src="https://github.com/user-attachments/assets/af37feb4-24b7-419a-9722-5db3238f73b4">
    <p><strong>Relationship Between Profile Visits and Followers Gained</strong></p>
</div>

- The scatter plot shows a linear relationship between profile visits and new followers, suggesting that driving more traffic to the profile can lead to increased follower growth.
## Conclusion


## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/crishN144/insta-analysis.git
   cd insta-analysis
   ```

2. **Set Up the Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Analysis**:
   - Open Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Navigate to and open `instagram_reach_analysis.ipynb`
   - Execute the cells to replicate the analysis.

4. **Explore and Adapt**:
   - Use the reach prediction model to estimate post performance.
   - Modify parameters and adapt the analysis to different datasets.

## Future Work

**Enhanced Sentiment Analysis**:
   - Implement sentiment analysis on user comments to identify common themes.
   - Explore correlations between sentiment scores and engagement metrics.
