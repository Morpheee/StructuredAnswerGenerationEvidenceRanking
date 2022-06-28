# StructuredAnswerGenerationEvidenceRanking
A Structure-Aware Model for Joint Answer Generation and Evidence Ranking in Long-Form Question Answering

## current state

The current version of the repo only proposes a baseline to the future proposed model.

The baseline is a combination of a retriever (DPR) and a generator (BART) trained separately and combined at inference in an end-to-end pipeline.

One can find 
* DPR retriever model in folder ./baseline/retriever 
* BART generator model in folder ./baseline/generator