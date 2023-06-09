""""""
import os
import uvicorn
from fastapi import FastAPI
from src.start import (classifier, 
                       pubs)
from src.config import logger
from src.data_types import (TemplateIds,
                            SearchData)
from src.utils import timeit


os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI(title="ExpertBotFastText")


@app.post("/api/search")
# @timeit
async def search(data: SearchData):
    """searching etalon by  incoming text"""
    if data.pubid in pubs:
        logger.info("searched text: {}".format(str(data.text)))
        try:
            logger.info("searched text without spellcheck: {}".format(str(data.text)))
            result = await classifier.searching(str(data.text), 0.4)
            return {"templateId": result["templateId"], "templateText": result["templateText"]}
        except Exception:
            logger.exception("Searching problem with text {} in pubid {}".format(str(data.text), str(data.pubid)))
            return {"templateId": 0, "templateText": ""}
    else:
        return {"templateId": 0, "templateText": ""}


if __name__ == "__main__":
    # uvicorn.run(app, host=service_host, port=service_port)
    uvicorn.run(app, host="0.0.0.0", port=8080)
