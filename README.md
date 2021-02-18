# RankIT---final-year-project
An automatic intrinsic evaluation system of English summaries.

Today in the era of information, information is everywhere. We are overwhelmed by it, in this era,
we can get every single piece of information from the internet: we can find books, articles, news.
But how can we obtain all this information, how can we get only the essential pieces from a given
text?
The answer is to summarize the text; in other words: condense the text into a smaller representation
that expresses the primary keys, facts, and ideas fluently.
Fortunately, there are plenty of automatic summarizers that can do the trick.
Automatic summarizers aim to produce a shorter but concise version of the original text.
The summarization process is not trivial. To produce a good summary,
the summarizer needs to understand the text, its semantics, and main ideas. Eventually, it needs to
produce an easy to understand, coherent piece of text.
There are two main summarization evaluation methods, intrinsic and extrinsic.
In the extrinsic methods, the summary gets its score regarding some tasks; this type of evaluation is
useful for goal-oriented summaries.
In the intrinsic methods, the summary directly analyzed. The intrinsic methods may involve a direct
comparison with the original text(by measuring the coverage of the main topics) or by the closeness
between the summary and some golden standard summaries that are written by human-experts.
The problem with the intrinsic methods that rely on golden standard is that they require much
workforce and lack consistency.
In this project, I built a system for automatic intrinsic evaluation of English summaries.
I have used a supervised approach in which I have trained deep learning models on pairs of (articles,
summaries) and a corresponding score in 0-1 range.
I have taken 16,000 articles from the CORD-19: The Covid-19 Open Research Dataset.
I have generated the summaries and the scores in the following way:
(1)Abstract of the article as 1 score (gold standard).(2) Random permutation of the summaries as 0
scores. (3)Concatenation of the first abstract half with noise from news domain, zeros, or duplication
of the first half.
I have tried the following deep learning models:
(1)Siamese network of two bi-directional LSTMs (with and without attention).
(2)Siamese network of multi-channeled CNN (to try and capture multiple local contexts),
(3)Siamese network of LSTMs on top of the Siamese network of CNN (to try and capture local
contexts alongside with global ones) and (4) the best of the above but with two parallel networks:
one for the score, second for the penalty(on duplicated content for Example).
I decided to use MSE cost function as my metric because the domain of the problem and because I
wanted to give "high penalties" to higher errors.
With 0.0035 (MSE) on the train set, 0.0247 on the dev set, and 0.0210 on the test set, the LSTM with
attention was the best. The results are not bad, but there were two main problems:(1) The network
failed to understand when the summary incomplete. (2) our model overfitted! I have tried to deal
with the first problem by duplicating the network into one that gives a score and one that gives
penalties, this approach needs further analysis.
Keywords: Deep Learning, Siamese Network, Metric Learning, Summarization Evaluation.
