# Content selection from the context with MaRGE
We propose a BERT-based regression model, heavily relying on the work presented in the paper https://arxiv.org/pdf/2012.14774.pdf. 

It measures the relevance score of a given sentence from the context w.r.t. the associated RDF triples.

At inference time, it computes the relevance score for each sentence in the context, ranks them and selects the top k sentences.
