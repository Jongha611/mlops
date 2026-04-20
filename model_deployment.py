import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BatchEncoding, BertTokenizer, BertForSequenceClassification
from schema import PredictRequest, PredictResponse
from torch import nn
from ray import serve


app = FastAPI()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5 if torch.cuda.is_available() else 0},
)
@serve.ingress(app)
class BertDeployment:

    def __init__(self):
        self.device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
        state_dict_path = "models/bert/bert_news_classifier_v2.pth"
        state_dict = torch.load(
            f=state_dict_path,
            weights_only=True,
            map_location=self.device,
        )
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", num_labels=4)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        self.label2id = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}

    @app.post(path="/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        batch: dict[list[str]] = request.model_dump()

        inputs = self.tokenizer(
            batch["data"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = nn.functional.softmax(
                input=outputs.logits,
                dim=1,
            )
            model_predictions: list[int] = probabilities.argmax(dim=1).tolist()
            confidences: list[float] = probabilities.max(dim=1).values.tolist()

            label_predictions: list[str] = [self.id2label[prediction] for prediction in model_predictions]

        return PredictResponse(
            label_predictions=label_predictions,
            confidences=confidences,
        )


deployment = BertDeployment.bind()