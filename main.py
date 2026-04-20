from fastapi import FastAPI

from ray import serve
from ray.serve.handle import DeploymentHandle

from app.schema import PredictRequest, PredictResponse

from app.ai_service import BertNewsClassifier
from app.preprocessor import TextPreprocessor


app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:

    def __init__(
        self, 
        text_preprocessor_handle: DeploymentHandle, 
        classification_model_handle: DeploymentHandle
    ):
        self.text_preprocessor = text_preprocessor_handle
        self.classification_model = classification_model_handle

    @app.post(path="/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        preprocessed_data = self.text_preprocessor.preprocess.remote(request.data)
        predict_response: PredictResponse = await self.classification_model.inference.remote(preprocessed_data)

        return predict_response


preprocessor_deployment = TextPreprocessor.bind()
model_deployment = BertNewsClassifier.bind()
deployment = APIIngress.bind(
    text_preprocessor_handle=preprocessor_deployment,
    classification_model_handle=model_deployment,
)

