import os
import compress_fasttext
import pandas as pd
import numpy as np
from src.texts_processing import TextsTokenizer
from src.config import (stopwords,
                    parameters,
                    logger,
                    PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier



ft_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(os.path.join(PROJECT_ROOT_DIR, 
                                                                          "models", 
                                                                          "geowac_tokens_sg_300_5_2020-100K-20K-100.bin"))


tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)

etalons_df = pd.read_csv(os.path.join("data", "etalons.csv"), sep="\t")
tokens_texts = tokenizer(list(etalons_df["query"]))
index = [ft_model.get_sentence_vector(lm_tx) for lm_tx in tokens_texts]
labels = [(lb, ans) for lb, ans in zip(etalons_df["label"], etalons_df["templateText"])]
answers_dict = {num: {"templateId": itm[0], "templateText": itm[1]} for num, itm in enumerate(labels)}

pubs_df = pd.read_csv(os.path.join("data", "pubs.csv"), sep="\t")
pubs = list(pubs_df["pubid"])

classifier = FastAnswerClassifier(index, ft_model, answers_dict, tokenizer)
logger.info("service started...")
