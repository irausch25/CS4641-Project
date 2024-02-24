---
layout: default
title: CS4641 Project Proposal
---

# Introduction/Background:

- **Introduction**: The project is based around using NLP for a more nuanced understanding of emotional context in text beyond positive, neutral, and negative classifications. Understanding emotional context in text has many applications such as customer feedback analysis, mental health assessment, and social media monitoring.
  
- **Literature review**: Pang and Lee [8] argue in their book that the field of sentiment analysis has evolved from broad classifications to detailed emotion detection due to the complexities of human language and sentiment. Deep learning, especially LSTM [6] and BERT [9], have revolutionized NLP, offering tools capable of a better nuanced understanding.

- **Dataset Description**: The "Emotions" dataset contains english twitter messages with corresponding predominant emotion conveyed using six fundamental emotions denoted by numbers: 
sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).


- Click [here](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) to view the dataset from Kaggle.

# Problem Definition:

- **Problem**: Classifying exact fine-grained emotions in text poses a challenge due to the subtle nuances in text. This is the problem that the project tries to solve by accurately classifying text into one of six emotion categories.

- **Motivation**: The motivation behind the project are the applications of understanding emotional context in text. Improved emotion detection can aid in mental health assessment, customer feedback analysis, and social media monitoring. This work is crucial for developing AI that can respond to human emotions more effectively and empathetically. 

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

| Name              | Proposal Contributions                                                                                            |
|:------------------|:------------------------------------------------------------------------------------------------------------------|
| Ian Rausch        | Github pages setup and updates, Results and Discussion, Project Goals section, Data Preprocessing Methods section |
| Parag Ambildhuke  | Github theme setup, Github pages updates, Introduction /Background section, Problem Definition section            |
| Pritesh Rajyaguru | Powerpoint Presentation, Video Recording                                                                          |
| Shubham Dhar      | Github pages updates, Models section                                                                              |
| Zachary Seletsky  | Powerpoint Presentation, Video Recording                                                                          |

# References:

[1] C. P. Chai, “Comparison of text preprocessing methods,” Natural Language Engineering, vol. 29, no. 3, pp. 509–553, 2023. doi:10.1017/S1351324922000213

[2] K. Machová, M. Szabóova, J. Paralič, and J. Mičko, “Detection of emotion by Text Analysis Using Machine Learning,” Frontiers in Psychology, vol. 14, Sep. 2023. doi:10.3389/fpsyg.2023.1190326

[3] P. Nandwani and R. Verma, “A review on sentiment analysis and emotion detection from text,” Social Network Analysis and Mining, vol. 11, no. 1, Aug. 2021. doi:10.1007/s13278-021-00776-6

[4] E. Batbaatar, M. Li, and K. H. Ryu, “Semantic-emotion neural network for emotion recognition from text,” IEEE Access, vol. 7, pp. 111866–111878, Aug. 2019. doi:10.1109/access.2019.2934529 

[5]A. Chatterjee, U. Gupta, M. K. Chinnakotla, R. Srikanth, M. Galley, and P. Agrawal, “Understanding Emotions in Text Using Deep Learning and Big Data,” Computers in Human Behavior, vol. 93, pp. 309–317, Apr. 2019, doi: https://doi.org/10.1016/j.chb.2018.12.029.  

[6] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: https://doi.org/10.1162/neco.1997.9.8.1735. 

[7] I. Albu and S. Spînu, "Emotion Detection From Tweets Using a BERT and SVM Ensemble Model," ArXiv.Org, 2022. Available: https://www.proquest.com/working-papers/emotion-detection-tweets-using-bert-svm-ensemble/docview/2700434497/se-2. 

[8] B. Pang and L. Lee, Opinion mining and sentiment analysis. Boston: Now Publishers, 2008.

[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, Oct. 11, 2018. https://arxiv.org/abs/1810.04805
