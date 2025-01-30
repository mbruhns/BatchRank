# Batch Effect Assessment using Ranking

BatchRank is a method for assessing batch effects in single-cell(biomedical) data. It is based on the idea that batch effects can be
detected by conducting a Classifier-2-Sample-Test (C2ST). BatchRank runs a C2ST for each of the provided target variables and ranks the
results. The ranking is then used to quantify the detected batch effects using any suitable classification metric (default is balanced
accuracy).
