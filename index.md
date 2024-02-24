---
layout: default
title: CS4641 Project Proposal
---

# Introduction/Background:

- **Introduction**: The project is based around using NLP for a more nuanced understanding of emotional context in text using 6 fundamental human emotions.
- **Literature review**: Pang and Lee [8] argue in their book that the field of sentiment analysis has evolved from broad classifications to detailed emotion detection due to the complexities of human language and sentiment. Deep learning, especially LSTM [6] and BERT [9], have revolutionized NLP, offering tools capable of a better nuanced understanding.
- **Dataset Description**: Contains english twitter messages with corresponding predominant emotion conveyed using six fundamental emotions denoted by numbers:
  sadness(0), joy(1), love(2), anger(3), fear(4), and surprise(5).

- Click [here](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) to view the dataset from Kaggle.

# Problem Definition:

- **Problem**: Classifying exact emotions in text poses a challenge due to the subtle nuances in text. The project tries to solve this problem by accurately classifying text into one of six emotion categories.

- **Motivation**: Improved emotion detection has applications like aid in mental health assessment, customer feedback analysis, and social media monitoring. It’s crucial for developing AI that can respond to human emotions more empathetically.

# Methods:

**Preprocessing**: Based on Chai [1], we have identified the following 3 data preprocessing methods:

- Text Normalization: Ensuring consistency by converting text to a uniform format (e.g., lowercase, removing extraneous characters and whitespace).
- Tokenization: Breaking down the sentences into individual tokens or words for efficient processing.
- Negation Handling: Identifying a set of words affected by negation, then appending a prefix, so that we can distinguish between good and not good. This is crucial as Chai [1] emphasizes “preserving negation is essential.. removing the negation term will result in an opposite meaning and ambiguity….”

These 3 methods will result in a clean and structured dataset that can be easily analyzed.

**Models/Algorithms**:

- Naïve Bayes: Used using Word-to-vec to get features along with Bayesian methods [10] by turning a sentence into a vector.
- SVM: Tokenize the sentences, normalize the features, and then run them through an SVM [11] to train it. Also, we can use the Kernel trick to add non-linearity.
- LSTM: Used to tokenize and then use Stochastic Gradient Descent(SGD) algorithm to train the LSTM [6]. Will require more hyperparameter tuning and resources than a traditional RNN.
- BERT: Take the dataset and split it into two parts: training-section and test-section. We fine tune BERT[7] on the training-section and then use the fine-tuned BERT to make predictions on the test-section.

# (Potential) Results and Discussion:

We will assess the performance of our models based on F1 score, accuracy, and precision with the relative performance ranking of the models being Naive Bayes, SVM, RNN, and BERT across all 3 quantitative measures.

These models can be expected to achieve the following results:

- Accuracy:
  - Naive Bayes: 74.1% [3]
  - SVM: 85%+ [3]
  - RNN: 85%+ [3]
  - BERT: 94% [3]
- Precision:
  - Naive Bayes: 50% [2]
  - SVM: 67% [2]
  - RNN: 95% [2]
  - BERT: 95%+
- F1 Score:
  - Naive Bayes: 0.49 [4]
  - SVM: 0.535 [4]
  - RNN: 0.54 [4]
  - BERT: 0.84 [7]

For expected results, we anticipate replicating the relative performance ranking from our goals; however, we expect lower overall scores due to differences in datasets and training parameters.

# Gnatt Chart:

[Download CSV](UpdatedGanttChart.xlsx)

# Contribution Table:

| Name              | Proposal Contributions                                                                                                                |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| Ian Rausch        | Github pages setup and updates, Results and Discussion, Project Goals section, Data Preprocessing Methods section, discussion with TA |
| Parag Ambildhuke  | Github theme setup, Github pages updates, Introduction /Background section, Problem Definition section, discussion with TA            |
| Pritesh Rajyaguru | Powerpoint Presentation, Video Recording, team meetings setup                                                                         |
| Shubham Dhar      | Github pages updates, Models section, discussion with TA                                                                              |
| Zachary Seletsky  | Powerpoint Presentation, Video Recording, discussion with TA                                                                          |

# References:

[1] C. P. Chai, “Comparison of text preprocessing methods,” Natural Language Engineering, vol. 29, no. 3, pp. 509–553, 2023. doi:10.1017/S1351324922000213

[2] K. Machová, M. Szabóova, J. Paralič, and J. Mičko, “Detection of emotion by Text Analysis Using Machine Learning,” Frontiers in Psychology, vol. 14, Sep. 2023. doi:10.3389/fpsyg.2023.1190326

[3] P. Nandwani and R. Verma, “A review on sentiment analysis and emotion detection from text,” Social Network Analysis and Mining, vol. 11, no. 1, Aug. 2021. doi:10.1007/s13278-021-00776-6

[4] E. Batbaatar, M. Li, and K. H. Ryu, “Semantic-emotion neural network for emotion recognition from text,” IEEE Access, vol. 7, pp. 111866–111878, Aug. 2019. doi:10.1109/access.2019.2934529

[5] A. Chatterjee, U. Gupta, M. K. Chinnakotla, R. Srikanth, M. Galley, and P. Agrawal, “Understanding Emotions in Text Using Deep Learning and Big Data,” Computers in Human Behavior, vol. 93, pp. 309–317, Apr. 2019, doi: https://doi.org/10.1016/j.chb.2018.12.029.

[6] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: https://doi.org/10.1162/neco.1997.9.8.1735.

[7] I. Albu and S. Spînu, "Emotion Detection From Tweets Using a BERT and SVM Ensemble Model," ArXiv.Org, 2022. Available: https://www.proquest.com/working-papers/emotion-detection-tweets-using-bert-svm-ensemble/docview/2700434497/se-2.

[8] B. Pang and L. Lee, Opinion mining and sentiment analysis. Boston: Now Publishers, 2008.

[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, Oct. 11, 2018. https://arxiv.org/abs/1810.04805

‌[10] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient Estimation of Word Representations in Vector Space,” arXiv.org, Sep. 07, 2013. https://arxiv.org/abs/1301.3781

[11] “Support vector machines - IEEE Journals & Magazine,” Ieee.org, 2019. https://ieeexplore.ieee.org/document/708428
