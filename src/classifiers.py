"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
from src.data_types import Parameters
from src.texts_processing import TextsTokenizer
from src.config import logger
import numpy as np

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(10)  # timeout


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, index, ft_model, answers_dict, tokenizer):
        self.index = index
        self.answers = answers_dict
        self.ft_model = ft_model
        self.tokenizer = tokenizer

    async def searching(self, text: str, score: float):
        """"""
        """searching etalon by  incoming text"""
        try:
            tokens = self.tokenizer([text])[0]
            if tokens:
                q_vc = self.ft_model.get_sentence_vector(tokens)
                cos_sims_arr = self.ft_model.cosine_similarities(q_vc, self.index)
                cos_sims = cos_sims_arr.tolist()
                best_score = max(cos_sims)
                if best_score and best_score >= score:
                    logger.info("search completed successfully with result: {}".format(str(self.answers[cos_sims.index(best_score)])))
                    d = self.answers[cos_sims.index(best_score)]
                    d["score"] = best_score
                    return d
                else:
                    logger.info("not found answer for input text {}".format(str(text)))
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return {"templateId": 0, "templateText": ""}

