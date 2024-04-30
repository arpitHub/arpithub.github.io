---
layout: post
title: Odds and Odds Ratio
subtitle: Understanding Odds, Odds Ratios, and Their Use in Medicine
cover-img: /assets/img/odds_ratio_cover.jpeg
thumbnail-img: /assets/img/odds_ratio_thumb.png
share-img: /assets/img/odds_ratio_main.jpeg
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [statistics,book]
comments: true
---

I'm currently reading `The Art of Statistics` by Sir David John Spiegelhalter and I'd like to discuss the concepts of odds and odds ratios. We encounter these terms frequently when discussing the likelihood of events, from the odds of winning a game to the effectiveness of medical treatments.

#### What are Odds?

Odds represent the probability of an event happening versus the probability of it not happening. They are calculated as:

```Odds of an event happening / Odds of an event not happening```

#### What is Odds Ratio?

An odds ratio compares the odds of an event happening under two different conditions. It's calculated as:

```(Odds of event happening in one group ) / (Odds of event happening in another group)```

#### Odds Ratios in Medicine

In medical research, odds ratios help us understand if a treatment, intervention, or exposure influences a particular health outcome.  Let's consider a vaccine campaign example:

#### Scenario:

**Intervention Group:** 500 people exposed to the vaccine campaign.
**Control Group:** 500 people not exposed to the campaign.
**Intervention Group Results:** 400 got vaccinated, 100 didn't.
**Control Group Results:** 275 got vaccinated, 225 didn't.

#### Calculating the Odds Ratio

1. **Odds in Intervention Group:** 400 (vaccinated) / 100 (not vaccinated) = 4
2. **Odds in Control Group:** 275 (vaccinated) / 225 (not vaccinated) ≈ 1.23
3. **Odds Ratio:** 4 / 1.23 ≈ 3.25

```python
import pandas as pd

# Create a sample dataset
data = {'Vaccinated': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
        'Group': ['Intervention', 'Intervention', 'Intervention', 'Intervention', 'Control', 'Control', 'Control', 'Control'],
        'Count': [400, 100, 275, 225, 275, 225, 275, 225]}
df = pd.DataFrame(data)

# Calculate odds for each group
df['Odds'] = df['Count'][df['Vaccinated'] == 'Yes'] / df['Count'][df['Vaccinated'] == 'No']

# Calculate the odds ratio
odds_ratio = df.loc[df['Group'] == 'Intervention', 'Odds'].iloc[0] / df.loc[df['Group'] == 'Control', 'Odds'].iloc[0]

print("Odds Ratio:", odds_ratio) 
```

#### Interpreting the Odds Ratio

- **Odds Ratio = 1:** The intervention (vaccine campaign) had no effect on the likelihood of getting vaccinated.
- **Odds Ratio > 1:** People exposed to the intervention were more likely to get vaccinated.
- **Odds Ratio < 1:** People exposed to the intervention were less likely to get vaccinated.

In our example, the odds ratio of 3.25 means people seeing the campaign were about 3 times more likely to get vaccinated than those who didn't. This suggests the campaign was effective.

#### Important Notes

Odds ratios, like correlation, show association but not necessarily causation.
Statistical tests (like Fisher's Exact Test, Chi-Square Test, or the Wald Test) can further confirm if the odds ratio shows a real effect.

```python
import pandas as pd
from scipy.stats import fisher_exact

# Create a sample dataset (with Control group)
data = {'Vaccinated': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
        'Group': ['Intervention', 'Intervention', 'Intervention', 'Intervention', 'Control', 'Control', 'Control', 'Control'],
        'Count': [400, 100, 275, 225, 275, 225, 275, 225]}
df = pd.DataFrame(data)

# Create a contingency table
contingency_table = pd.crosstab(df['Group'], df['Vaccinated'])
print(contingency_table)

# Perform Fisher's Exact Test
oddsratio, pvalue = fisher_exact(contingency_table)

print("Odds Ratio:", oddsratio)
print("p-value:", pvalue)
```

#### Conclusion
 Odds and odds ratios provide a straightforward way to understand the likelihood of events and the strength of associations. They help us quantify how much more (or less) likely something is to happen under different conditions. In fields like medicine, this understanding is invaluable. Researchers can use these concepts to evaluate the effectiveness of treatments, explore risk factors for diseases, and make informed decisions that ultimately improve health outcomes.  While it's important to employ appropriate statistical tests and avoid jumping to conclusions about causation, odds and odds ratios are essential tools in the ongoing pursuit of data-driven insights.

##### References:
1. [NCCMT - URE - Odds Ratios](https://www.youtube.com/watch?v=5zPSD_e_N04)
2. [StatQuest - Odds Ratios and Log(Odds Ratios), Clearly Explained!](https://www.youtube.com/watch?v=8nm0G-1uJzA)
