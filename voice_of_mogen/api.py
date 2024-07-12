import logging
import os

from flask import Flask, url_for
from flask import request
from flask_cors import CORS
from google.cloud import storage
from resemble import Resemble

from voice_of_mogen import Composition
from voice_of_mogen import Sentiment
from voice_of_mogen import VoiceOfMogen
from voice_of_mogen.audio import Audio
from voice_of_mogen.ssml import augment_text
from voice_of_mogen.utils import decode_audio
from voice_of_mogen.utils import encode_audio
from voice_of_mogen.utils import get_env
from voice_of_mogen.utils import make_fake_name
from voice_of_mogen.utils import set_adc
from voice_of_mogen.utils import upload_audio_to_gcs

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def phoneme_timestamps_to_hit_points(phoneme_timestamps, phoneme_characters):
    hit_points = []
    word_active = False

    for i, (char, times) in enumerate(zip(phoneme_characters, phoneme_timestamps)):
        if not word_active and char != " ":
            word_active = True
            hit_points.append(times[0])
        elif word_active and char == " ":
            word_active = False
    hit_points.append(phoneme_timestamps[-1][-1])

    return hit_points


def characters_to_text(characters):
    return "".join(characters)


def create_app():
    # setup GCP
    gcp_config = get_env()
    set_adc(gcp_config)
    storage_client = storage.Client(project=gcp_config["GOOGLE_PROJECT"])
    cache_bucket = storage_client.bucket(gcp_config["CACHE_BUCKET"])

    sample_rate = int(os.getenv("SAMPLE_RATE", default=44100))

    banana_api_key = os.getenv("BANANA_API_KEY", default=None)
    composition_endpoint = os.getenv("COMPOSITION_ENDPOINT", default=None)
    sentiment_endpoint = os.getenv("SENTIMENT_ENDPOINT", default=None)
    vocoder_endpoint = os.getenv("VOCODER_ENDPOINT", default=None)
    composition_model_key = os.getenv("COMPOSITION_MODEL_KEY", default=None)
    sentiment_model_key = os.getenv("SENTIMENT_MODEL_KEY", default=None)
    vocoder_model_key = os.getenv("VOCODER_MODEL_KEY", default=None)
    resemble_api_key = os.environ.get("RESEMBLE_API_KEY", default=None)
    resemble_voice_uuid = os.environ.get("RESEMBLE_VOICE_UUID", default=None)
    resemble_project_uuid = os.environ.get("RESEMBLE_PROJECT_UUID", default=None)
    print("RESEMBLE_API_KEY", resemble_api_key)
    Resemble.api_key(resemble_api_key)

    log.info(f"Creating Flask app at {__name__}...")
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    log.info("Done.")
    log.info(sample_rate)
    log.info(type(sample_rate))
    log.info("Initialising VoiceOfMogen...")
    voice_of_mogen = VoiceOfMogen()

    if composition_endpoint is not None and composition_model_key is not None:
        log.info("Using Banana Composer")
        voice_of_mogen = voice_of_mogen.with_composer(
            "banana",
            endpoint_url=composition_endpoint,
            api_key=banana_api_key,
            model_key=composition_model_key,
        )
    else:
        # log.info("Using Music Transformer Composer")
        # voice_of_mogen = voice_of_mogen.with_composer(
        #     "music-transformer",
        # )

        voice_of_mogen = voice_of_mogen.with_composer(
            "api", 
            endpoint_url=composition_endpoint
        )
    if sentiment_endpoint is not None and sentiment_model_key is not None:
        log.info("Using Banana Sentiment Analyzer")
        voice_of_mogen = voice_of_mogen.with_sentiment_analyzer(
            "banana",
            endpoint_url=sentiment_endpoint,
            api_key=banana_api_key,
            model_key=sentiment_model_key,
        )
    else:
        # log.info("Using XLM-R Sentiment Analyzer")
        # voice_of_mogen = voice_of_mogen.with_sentiment_analyzer(
        #     "xlm-roberta",
        # )

        voice_of_mogen = voice_of_mogen.with_sentiment_analyzer(
            "api", 
            endpoint_url=sentiment_endpoint
        )

    if vocoder_endpoint is not None and vocoder_model_key is not None:
        log.info("Using Banana Vocoder")
        voice_of_mogen = voice_of_mogen.with_vocoder(
            "banana",
            endpoint_url=vocoder_endpoint,
            api_key=banana_api_key,
            model_key=vocoder_model_key,
        )
    else:
        log.info("Using PSOLA Vocoder")
        voice_of_mogen = voice_of_mogen.with_vocoder(
            "psola",
        )

    voice_of_mogen = voice_of_mogen.init()
    log.info("Done.")

    open_requests = {}

    @app.route("/hello", methods=["POST", "GET"])
    def hello():
        return "hello"
    
    @app.route("/create_clip", methods=["POST", "GET"])
    def create_clip():
        if request.method == "GET":
            text = request.args.get("text")
            synthesis_parameters = request.args.get("synthesis_parameters", None)
            composition_parameters = request.args.get("composition_parameters", None)
            context = request.args.get("context", None)
        else:
            text = request.json.get("text")
            synthesis_parameters = request.json.get("synthesis_parameters", None)
            composition_parameters = request.json.get("composition_parameters", None)
            context = request.json.get("context", None)

        if (
            ## I modified this part
            synthesis_parameters is not None
            and "speech_parameters" in synthesis_parameters
            and synthesis_parameters["speech_parameters"] is not None
        ):
            text = augment_text(text, **synthesis_parameters["speech_parameters"])

        request_url = url_for('return_clip', _external=True, _scheme='https')
        print(request_url)
        voice = Resemble.v2.clips.create_async(
            resemble_project_uuid,
            resemble_voice_uuid,
            request_url,
            text,
        ## I modified this part
            sample_rate=int(sample_rate),
            include_timestamps=True,
        )
        ## I modified this part
        clip_uuid = voice["item"]["uuid"]
        open_requests[clip_uuid] = {
            "synthesis_parameters": synthesis_parameters,
            "composition_parameters": composition_parameters,
            "context": context,
            "status": "resemble",
        }

        return voice

    @app.route("/return_clip", methods=["POST"])
    def return_clip():
        clip_uuid = request.json["id"]
        if clip_uuid not in open_requests:
            return "Clip not found", 404

        synthesis_parameters = open_requests[clip_uuid]["synthesis_parameters"]
        composition_parameters = open_requests[clip_uuid]["composition_parameters"]
        context = open_requests[clip_uuid]["context"]
        open_requests[clip_uuid]["status"] = "resemble_done"

        clip_url = request.json["url"]
        phoneme_characters = request.json["audio_timestamps"]["phon_chars"]
        phoneme_timestamps = request.json["audio_timestamps"]["phon_times"]
        grapheme_characters = request.json["audio_timestamps"]["graph_chars"]

        length = phoneme_timestamps[-1][-1]
        audio_clip = Audio(
            clip_url=clip_url,
            duration_in_seconds=length,
            sample_rate=int(sample_rate),
            ## I modified this part
            normalize=False # synthesis_parameters["speech_parameters"].get("normalize", False),
        )

        hit_points = phoneme_timestamps_to_hit_points(
            phoneme_timestamps, phoneme_characters
        )
        text = characters_to_text(grapheme_characters)

        voice, sentiment, composition = voice_of_mogen.process(
            audio_clip,
            text,
            hit_points,
            context=context,
            synthesis_parameters=synthesis_parameters,
            composition_parameters=composition_parameters,
        )

        open_requests[clip_uuid]["status"] = "mogen_done"

        file_name = make_fake_name(16, prefix=clip_uuid)
        audio_url = upload_audio_to_gcs(
            cache_bucket,
            f"mogen-voice-outputs/{file_name}",
            voice.audio_data,
            int(sample_rate),
        )

        open_requests[clip_uuid]["audio_url"] = audio_url
        open_requests[clip_uuid]["status"] = "ready"
        print(f"Uploaded audio to {audio_url}")

        return {
            "audio_url": audio_url,
            "tail_start_in_samples": voice.metadata["tail_start_in_samples"],
            "sentiment": sentiment.to_dict(),
            "composition": composition.to_dict(),
        }

    @app.route("/get_clip")
    def get_clip():
        clip_uuid = request.args.get("id")

        if clip_uuid in open_requests and open_requests[clip_uuid]["status"] == "ready":
            return {"url": open_requests[clip_uuid]["audio_url"], "status": "ready"}

        if clip_uuid in open_requests:
            return {"status": open_requests[clip_uuid]["status"]}

        if clip_uuid not in open_requests:
            return "Clip not found", 404

        return "Unknown error", 500
        # possible_blobs = cache_bucket.list_blobs(
        #     prefix="mogen-voice-outputs", match_glob=f"{clip_uuid}*.wav"
        # )
        # possible_blobs = list(possible_blobs)
        #
        # if len(possible_blobs) == 0 and clip_uuid not in open_requests:
        #     return "Clip not found", 404
        # elif len(possible_blobs) == 0 and clip_uuid in open_requests:
        #     return {"status": open_requests[clip_uuid]["status"]}
        # elif len(possible_blobs) > 1:
        #     return "Multiple clips found", 500
        #
        # blob = possible_blobs[0]
        # del open_requests[clip_uuid]
        # return {"url": blob.public_url, "status": "ready"}

    @app.route("/list_clips")
    def list_clips():
        # list last N audio clips in the cache bucket. if they have a public URL,
        # link it. otherwise, link to /download_clip?id=<blob> to allow download

        blobs = cache_bucket.list_blobs(prefix="mogen-voice-outputs")

        list = "<ul>"
        for b in blobs:
            url = b.public_url
            list += f"<li><a href={url}>{url}</a></li>"
        list += "</ul>"
        return list

    @app.route("/analyze_sentiment", methods=["POST"])
    def analyze_sentiment():
        log.info(f"Received request {request.json}")
        text = request.json["text"]
        log.info(f"Analyzing sentiment for text: {text}")
        sentiment = voice_of_mogen.sentiment_analyzer.analyze(text)

        response = {
            "sentiment": {"valence": sentiment.valence, "arousal": sentiment.arousal}
        }
        return response

    @app.route("/compose", methods=["POST"])
    def compose():
        log.info(f"Received request {request.json}")
        log.info("Composing music...")
        if request.json["context"] is None:
            context = None
        else:
            context = Composition.from_dict(request.json["context"])

        composition = voice_of_mogen.composer.compose(
            Sentiment.from_dict(request.json["sentiment"]),
            request.json["duration_in_seconds"],
            request.json["timestamps"],
            context=context,
        )

        response = {"composition": composition.to_dict()}
        return response

    @app.route("/vocode", methods=["POST"])
    def vocode():
        log.info(f"Received request {request.json}")
        log.info("Vocoding...")
        audio = decode_audio(request.json["speech_audio"])
        composition = Composition.from_dict(request.json["composition"])
        synthesis_parameters = request.json["synthesis_parameters"]
        output, tail_idx = voice_of_mogen.vocoder.process(
            audio, composition, synthesis_parameters
        )

        response = {"audio": encode_audio(output), "tail_idx": tail_idx}
        return response

    return app
