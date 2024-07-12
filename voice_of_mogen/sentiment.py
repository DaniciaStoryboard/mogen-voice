import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import NamedTuple
from typing import Type

import requests
import torch
from banana_dev import Client as BananaClient
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizerFast

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Sentiment(NamedTuple):
    valence: float
    arousal: float

    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal}

    @staticmethod
    def from_dict(d: dict) -> "Sentiment":
        return Sentiment(d["valence"], d["arousal"])

    def __quantize(self, value: float) -> int:
        if value < -1 / 3:
            return -1.0
        elif value > 1 / 3:
            return 1.0
        else:
            return 0.0

    @property
    def quantized_valence(self) -> int:
        return self.__quantize(self.valence)

    @property
    def quantized_arousal(self) -> int:
        return self.__quantize(self.arousal)


class ISentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Sentiment:
        pass


class TransformerSentimentAnalyzer(ISentimentAnalyzer):
    def __init__(
        self,
        model: Type[PreTrainedModel],
        tokenizer: Type[PreTrainedTokenizer],
        checkpoint_path: str,
        device: str = "cpu",
    ):
        log.info(
            f"({self.__class__.__name__}) Loading sentiment analyzer from "
            f"{checkpoint_path}"
        )
        self.tokenizer = tokenizer.from_pretrained(checkpoint_path)
        log.info(
            f"({self.__class__.__name__}) Loading tokenizer from {checkpoint_path}"
        )
        self.model = model.from_pretrained(checkpoint_path, num_labels=2)
        self.model.eval()
        self.model.to(device)

        self.device = device

    def _postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def analyze(self, text: str) -> Sentiment:
        log.info(f"({self.__class__.__name__}) Analyzing sentiment for text: {text}")
        log.info(f"({self.__class__.__name__}) Tokenizing text...")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)
        log.info(f"({self.__class__.__name__}) Running model inference...")
        logits = self.model(**inputs).logits
        log.info(f"({self.__class__.__name__}) Postprocessing logits...")
        logits = self._postprocess(logits)

        valence = logits[0][0].item()
        arousal = logits[0][1].item()
        log.info(
            f"({self.__class__.__name__}) Text has valence: {valence:.3f}, "
            f"arousal: {arousal:.3f}."
        )
        return Sentiment(valence, arousal)


class DistilBertSentimentAnalyzer(TransformerSentimentAnalyzer):
    def __init__(
        self, checkpoint_path: str = "checkpoints/distilbert", tanh_scale: float = 5.0
    ):
        super().__init__(
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            checkpoint_path,
        )
        self.tanh_scale = tanh_scale

    def _postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits * self.tanh_scale)


class XLMRobertaSentimentAnalyzer(TransformerSentimentAnalyzer):
    def __init__(
        self, checkpoint_path: str = "checkpoints/xlm-roberta", device: str = "cpu"
    ):
        super().__init__(
            XLMRobertaForSequenceClassification,
            XLMRobertaTokenizerFast,
            checkpoint_path,
            device,
        )

    def _postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits)


class APISentimentAnalyzer(ISentimentAnalyzer):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def analyze(self, text: str) -> Sentiment:
        request = {"text": text}

        response = requests.post(self.endpoint_url, json=request)
        response.raise_for_status()

        sentiment_json = response.json()["sentiment"]
        return Sentiment.from_dict(sentiment_json)


class BananaSentimentAnalyzer(ISentimentAnalyzer):
    def __init__(self, endpoint_url: str, api_key: str, model_key: str):
        self.client = BananaClient(
            model_key=model_key,
            api_key=api_key,
            url=endpoint_url,
        )

    def analyze(self, text: str) -> Sentiment:
        request = {"text": text}
        response, meta = self.client.call("/analyze_sentiment", request)

        return Sentiment.from_dict(response["sentiment"])
