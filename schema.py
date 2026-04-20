from pydantic import BaseModel, Field


class PredictRequest(BaseModel):

    data: list[str] = Field(description="배치 설계를 위해 키의 값이 리스트로 되어있는 것으로 추측됨")


class PredictResponse(BaseModel):
    
    label_predictions: list[str] = Field(description="모델이 예측한 정답 (배치)")
    confidences: list[float] = Field(description="소프트맥스를 거친 모델 확신 확률 (배치)")

