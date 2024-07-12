import logging
from typing import List
from typing import Literal
from typing import Optional

from voice_of_mogen import Composition
from voice_of_mogen.audio import Audio
from voice_of_mogen.compose import APIComposer
from voice_of_mogen.compose import BananaComposer
from voice_of_mogen.compose import IComposer
from voice_of_mogen.compose import MusicTransformerComposer
from voice_of_mogen.sentiment import APISentimentAnalyzer
from voice_of_mogen.sentiment import BananaSentimentAnalyzer
from voice_of_mogen.sentiment import ISentimentAnalyzer
from voice_of_mogen.sentiment import XLMRobertaSentimentAnalyzer
from voice_of_mogen.vocode import APIVocoder
from voice_of_mogen.vocode import BananaVocoder
from voice_of_mogen.vocode import IVocoder
from voice_of_mogen.vocode import PSOLAVocoder

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class VoiceOfMogenInstance:
    def __init__(
        self,
        sentiment_analyzer: ISentimentAnalyzer,
        composer: IComposer,
        vocoder: IVocoder,
    ):
        self.sentiment_analyzer = sentiment_analyzer
        self.composer = composer
        self.vocoder = vocoder

    def process(
        self,
        speech_audio: Audio,
        text: str,
        timestamps: Optional[List[float]] = None,
        context: Optional[Composition] = None,
        composition_parameters: Optional[dict] = None,
        synthesis_parameters: Optional[dict] = None,
    ) -> Audio:
        log.info(f"Processing text input: {text}")

        log.info(f"Speech duration: {speech_audio.duration_in_seconds} seconds")

        log.info("Analyzing sentiment...")
        sentiment = self.sentiment_analyzer.analyze(text)

        log.info("Composing music...")
        composition = self.composer.compose(
            sentiment,
            speech_audio.duration_in_seconds,
            timestamps,
            context=context,
            composition_parameters=(composition_parameters or dict()),
        )

        log.info("Vocoding...")
        audio = self.vocoder.process(speech_audio, composition, synthesis_parameters)

        return audio, sentiment, composition


class VoiceOfMogen:
    def __init__(self):
        log.info("Creating VoiceOfMogen instance")
        self.sentiment_analyzer = None
        self.composer = None
        self.vocoder = None

    def with_sentiment_analyzer(
        self,
        sentiment_analyzer: Literal["xlm-roberta", "api", "banana"] = "xlm-roberta",
        **kwargs,
    ) -> "VoiceOfMogen":
        if sentiment_analyzer == "xlm-roberta":
            log.info("Using XLM-Roberta sentiment analyzer")
            self.sentiment_analyzer = XLMRobertaSentimentAnalyzer(**kwargs)
        elif sentiment_analyzer == "api":
            log.info("Using API sentiment analyzer")
            self.sentiment_analyzer = APISentimentAnalyzer(**kwargs)
        elif sentiment_analyzer == "banana":
            log.info("Using Banana sentiment analyzer")
            self.sentiment_analyzer = BananaSentimentAnalyzer(**kwargs)
        else:
            raise ValueError(f"Unknown sentiment analyzer: {sentiment_analyzer}")
        return self

    def with_composer(
        self,
        composer: Literal["music-transformer", "api", "banana"] = "music-transformer",
        **kwargs,
    ) -> "VoiceOfMogen":
        if composer == "music-transformer":
            log.info("Using MusicTransformer composer")
            self.composer = MusicTransformerComposer(**kwargs)
        elif composer == "api":
            log.info("Using API composer")
            self.composer = APIComposer(**kwargs)
        elif composer == "banana":
            log.info("Using Banana composer")
            self.composer = BananaComposer(**kwargs)
        else:
            raise ValueError(f"Unknown composer: {composer}")
        return self

    def with_vocoder(
        self,
        vocoder: Literal["psola", "api"] = "psola",
        **kwargs,
    ) -> "VoiceOfMogen":
        if vocoder == "psola":
            log.info("Using PSOLA vocoder")
            self.vocoder = PSOLAVocoder(**kwargs)
        elif vocoder == "api":
            log.info("Using API vocoder")
            self.vocoder = APIVocoder(**kwargs)
        elif vocoder == "banana":
            log.info("Using Banana vocoder")
            self.vocoder = BananaVocoder(**kwargs)
        else:
            raise ValueError(f"Unknown vocoder: {vocoder}")
        return self

    def with_defaults(self) -> "VoiceOfMogen":
        log.info("Building VoiceOfMogen with default components")
        self.with_sentiment_analyzer()
        self.with_composer()
        self.with_vocoder()
        return self

    def init(self) -> VoiceOfMogenInstance:
        log.info("Initializing VoiceOfMogen instance")
        if self.sentiment_analyzer is None:
            raise ValueError("Sentiment analyzer is not set")
        if self.composer is None:
            raise ValueError("Composer is not set")
        if self.vocoder is None:
            raise ValueError("Vocoder is not set")

        return VoiceOfMogenInstance(
            sentiment_analyzer=self.sentiment_analyzer,
            composer=self.composer,
            vocoder=self.vocoder,
        )
