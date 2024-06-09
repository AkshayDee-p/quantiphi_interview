# quantiphi_interview
Assignment solution to interview question

Assumptions :- End to end working RAG module only

List of issues :- 
 Currently the pdf reader does not read the data correctly so the embeddings generated would not be proper and the retreival would be having a very low accuracy.

    --> To solve this the data read from the pdf should be further cleaned and then converted into vector embeddings. [ Gain of Accuracy ]
    --> The chunking logic value is randomly selected , If we really want a better performance , the dataset needs to be studied and a custom logic for chunking should be built according to the context, as well as appropiate overlap should           exist to make the final retrieval of context better.
    --> A better sentence embedder should be used as currently the sentence transformer used here has a semantic comparision accuracy of 60%. [https://sbert.net/docs/sentence_transformer/pretrained_models.html]

After  we solve the above issue we can move on to solving more advance issues which can be further solved by building an information ranking system for getting the best context for a given user query for the llm to answer the question

  
# basic_implementation.py

The solution has been built with bare minimum usage of packages , most of the internal workings have been built in-house

# langchain_implementation

This solution has used a lot of langchain based packages for solving this particular issue.
