# Clean_meat_topic_modeling
## What did you say, clean meat? 
- Topic modeling and analyzing the sentiment of Reddit comments on clean meat, aka lab-grown meat
### Project 4 at Metis

### The goal of this project is utilize unsupervised learning techniques to identify topics on Reddit comments for "clean meat" and lab-grown" meat.
Data obtained from scraping Reddit for posts on "clean meat" and "lab-grown meat".

Contents:
1. [PART 1 Scraping Reddit](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Project4_PART1_Scrape_Reddit.ipynb)
2. [PART 2 Cleaning Reddit Data](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Project4_PART2_Cleaning_Reddit_Data.ipynb)
3. [PART 3 Final model for lab-grown meat](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Project4_PART3_FinalModel_LabGrown.ipynb)
4. [PART 4 Final model for clean meat](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Project4_PART4_FinalModel_CleanMeat.ipynb)
5. [PART 5 Plot amount of comments over time](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Project4_PART5-Amount_comments_over_time.ipynb)
6. Used [helper functions](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/helper_functions.py)

### Background
In the past decade, scientists have been working on creating meat outside of the animal. Eliminating the need for animals for food production can potentially decrease environmental impacts, promote animal welfare, and increase food safety. 

To put it simply, lab-grown meat is basically taking animal stem cells and propagating them outside of the animal, like in a bioreactor. A bioreactor can be thought of as a large vessel where yogurt or beer is made. 

However, there is a negative connotation when it comes to lab-grown products. Many are hestiant towards trying or adopting lab-grown meat in their diet because of the mysteriousness or weirdness factor in the label. As a result, the industry wants to label the product as "clean meat."

The goal of this project it to conduct topic modeling on Reddit comments. With an LDA model, the coherence score was around 0.55 for both "lab-grown meat" and "clean meat" model. 

Below are selected word clouds as an example of the result from topic modeling.

Lab-grown meat: cost
Sentiment: 0.030, Proportion of positive comments = 25.3%
![Cost](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Final_Wordclouds/wordcloud_9_top1.png)

Clean meat: cost
Sentiment: 0.141, Proportion of positive comments: 50.9%
![Cost](https://github.com/chennat811/Clean_meat_topic_modeling/blob/main/Final_Wordclouds/wordcloud_clean_top1.png)

It seems like there is slight skeptism when it comes to the cost of the product. For the label "lab-grown meat", the sentiment score was lower than for "clean meat".

### Tools Used
Pandas, Matplotlib, Sklearn, Gensim, NLTK, Reddit API, wordcloud

### Impacts
Topic modeling and sentiment analysis can help the industry determine what the consumers are interested in. They can respond to consumers by releasing content that sparks interested. They can also identify the most negative pieces of comments and repond to those. If there are fake or outdated news that generate a lot of talk, the industry can identify those and respond to it. Future work: To automatically identify questions that the industry can answer and neutralize negative comments. Create a bot that answers those questions with scientific evidence.
