# Problem statement:
We have text, we want to infer an internal human state from it. A latent variable. We want an emotional direction or resolution behind the words.

-------

# What makes this hard:
Texts are ambiguous, contextual and indirect. We don't always say "I am experiencing positive effect". We say "This project saved my sanity."
This contants no explicit "good" word, no smiley, no score. The task is inference, not detection.
Machines like patterns, they don't like inference.

The baseline thinking is a caveman algorithm. It is rule based.
- if text contains "good" -> positive
- if text contains "bad" -> negative

But this fails instantly. So we upgrade.

-------

# The first real leap
We do a little trick. We assign similar numbers to texts with similar meanings.
The simplest version of this is:
- count words
- weight rare words more

and that's TF-IDF.

TF is term frequency. TF(word, document) asks how many times does this word appear in this document? For example, in "This project saved my sanity", every word has the same word count of 1. So TF  says all words are equal, which is obviously wrong. So we fix it.

IDF is Inverse Document Frequency. IDF(words) asks does this word appread in every document, or only a few? Words like "the", "this", "is" appear everywhere, hench less informative. Words like "sanity", "brilliant", "garbahe" appear less often and hence more informative.

Mathematically: IDF(word)= log(total_documents/documents_containing_word)

SO, common words have low IDF, rare words have high IDF.

Now we combine both the ideas. TF * IDF. A word is impotaint if (It appears often in this document) AND (It does NOT appear everywhere).
So every sentence becomes a vector like: [0,0,0,0,1.23,0,0,0,...]. Each number= TF-IDF score of a word. 

-------

# A classifier (decision-making layer)

Once text is numbers, we ask 'Given these numbers, which side of a boundary does this example fall on?'
That's Logistic Regression and SVM.

These models draw decision boundaries and spearate negative vs positive regions. They are linear thinkers.

---------

# Context enters the game (transformers)

Instead of treating words independently, we say "meaning depends on surrounding words."
Transformers do this by:
- reading the entire sentence at once
- letting each words attend to every other word
- building a contextual representation

so "sanity" near "saved" is not equal to "sanity" near "lost". That's the breakthrough.

Transformers do not output "sentiment". They output dense vectors and represent the meaning of the whole sentence. Then we add a small classifier head on top.
So the architecture is:
Text
 → Tokens
 → Transformer (context understanding)
 → Sentence embedding
 → Classifier head
 → Sentiment label

------------

# Tokenization

