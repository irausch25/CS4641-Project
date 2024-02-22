---
layout: default
title: CS4641 Project Proposal
---

# Introduction/Background:

- **Literature review**: Understanding emotional context in text has many applications such as customer feedback analysis, mental health assessment, and social media monitoring. The evolution of NLP has enabled more nuanced classifications beyond positive, neutral, and negative sentiments.

- **Dataset Description**: The "Emotions" dataset we chose contains English Twitter messages annotated with corresponding predominant emotion conveyed using six fundamental emotions: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
  This dataset is ideal for sentiment analysis and emotion classification tasks by analyzing the diverse spectrum of emotions expressed in short-form text on social media.

- Click [here](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) to view the dataset from Kaggle.

# Problem Definition:

- **Problem**: While sentiment analysis on text data has been widely explored in the past, classifying fine-grained emotions poses a challenge due to the subtle nuances in text. The problem is to accurately classify text into one of six emotional categories.

- **Motivation**: Improved emotion detection can aid in mental health assessment, customer feedback, and social engagement by providing nuanced insights beyond basic sentiment analysis. This work is crucial for developing AI that can respond to human emotions more effectively and empathetically.

# Methods:

**Preprocessing**:

- Text Normalization: For converting text to a uniform format, such as lowercasing all letters and removing special characters or unnecessary whitespace.
- Tokenization: Breaking down the sentences in our dataset into individual tokens or words.
- Negation Handling: Identifying a set of words affected by negation, then appending a prefix, so that we can distinguish between good and not good.
  - “Preserving negation is essential to the survey text, because removing the negation term will result in an opposite meaning and ambiguity…” [1]

**Models/Algorithms**:

- Naive Bayes
- Decision Trees
- RNNs
- Support Vector Machine (SVM)
- BERT

# (Potential) Results and Discussion:

TBD

# Gnatt Chart:

[Download CSV](GanttChart.xlsx)

# Contribution Table:

| Name       | Proposal Contributions |
| :--------- | :--------------------- |
| Ian Rausch | good swedish fish      |
| Parag      | good and plenty        |
| Pritesh    | good `oreos`           |
| Shube      | good `zoute` drop      |
| Zachary    | good `zoute` drop      |

# References:

[1] C. P. Chai, “Comparison of text preprocessing methods,” Natural Language Engineering, vol. 29, no. 3, pp. 509–553, 2023. doi:10.1017/S1351324922000213

[2] K. Machová, M. Szabóova, J. Paralič, and J. Mičko, “Detection of emotion by Text Analysis Using Machine Learning,” Frontiers in Psychology, vol. 14, Sep. 2023. doi:10.3389/fpsyg.2023.1190326

[3] P. Nandwani and R. Verma, “A review on sentiment analysis and emotion detection from text,” Social Network Analysis and Mining, vol. 11, no. 1, Aug. 2021. doi:10.1007/s13278-021-00776-6
