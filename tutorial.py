# import argparse
# import base64
# import json

# from googleapiclient import discovery
# import httplib2
# from oauth2client.client import GoogleCredentials


# DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
#                  'version={apiVersion}')


# def get_speech_service():
#     credentials = GoogleCredentials.get_application_default().create_scoped(
#         ['https://www.googleapis.com/auth/cloud-platform'])
#     http = httplib2.Http()
#     credentials.authorize(http)

#     return discovery.build(
#         'speech', 'v1beta1', http=http, discoveryServiceUrl=DISCOVERY_URL)


# def main(speech_file):
#     """Transcribe the given audio file.

#     Args:
#         speech_file: the name of the audio file.
#     """
#     with open(speech_file, 'rb') as speech:
#         speech_content = base64.b64encode(speech.read())

#     service = get_speech_service()
#     service_request = service.speech().syncrecognize(
#         body={
#             'config': {
#                 'encoding': 'LINEAR16',  # raw 16-bit signed LE samples
#                 'sampleRate': 16000,  # 16 khz
#                 'languageCode': 'en-US',  # a BCP-47 language tag
#             },
#             'audio': {
#                 'content': speech_content.decode('UTF-8')
#                 }
#             })
#     response = service_request.execute()
#     print(json.dumps(response))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         'speech_file', help='Full path of audio file to be recognized')
#     args = parser.parse_args()
#     main(args.speech_file)

#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the REST API for batch
processing.
Example usage:
    python transcribe.py resources/audio.raw
    python transcribe.py gs://cloud-samples-tests/speech/brooklyn.flac
"""

# [START import_libraries]
import argparse
import io
# [END import_libraries]


def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    speech_client = speech.Client()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
        audio_sample = speech_client.sample(
            content=content,
            source_uri=None,
            encoding='LINEAR16',
            sample_rate_hertz=16000)

    alternatives = audio_sample.recognize('en-US')
    for alternative in alternatives:
        print('Transcript: {}'.format(alternative.transcript))


def transcribe_gcs(gcs_uri):
    """Transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech
    speech_client = speech.Client()

    audio_sample = speech_client.sample(
        content=None,
        source_uri=gcs_uri,
        encoding='FLAC',
        sample_rate_hertz=16000)

    alternatives = audio_sample.recognize('en-US')
    for alternative in alternatives:
        print('Transcript: {}'.format(alternative.transcript))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='File or GCS path for audio file to be recognized')
    args = parser.parse_args()
    if args.path.startswith('gs://'):
        transcribe_gcs(args.path)
    else:
        transcribe_file(args.path)