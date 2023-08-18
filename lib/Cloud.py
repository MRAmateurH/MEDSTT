from google.cloud import storage
from google.cloud import texttospeech
from google.cloud import speech

from pydub import AudioSegment
from pydub.playback import play

# from lib.log import log
import lib.log as log

import io
import os
import time

# Google Cloud Storage Class Blob:

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "civil-orb-382813-63b889810cff.json"


def combinewav(original_audio, audio_content):
    output_audio = AudioSegment.from_wav(original_audio)
    new_audio = AudioSegment.from_file(io.BytesIO(audio_content), format='wav')
    output_audio = output_audio + new_audio

    return output_audio

class Storage:

    def __init__(self, credentials: str) -> None:
        # Change to your "GOOGLE_APPLICATION_CREDENTIALS"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
        storage_client = storage.Client()

    
    def upload_to_bucket(self, blob_name, file_path, bucket_name):
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            return True
        except Exception as e:
            print(e)
            return False
        
class TxtToSpeech:

    def __init__(self, credentials: str) -> None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
        self.tts_client = texttospeech.TextToSpeechClient()

    def txttospeech(self, contents: str, output_file: str):
        '''
        '''
        # output_directory = f'./audio/8000/dataset981'
        # output_file = f'./audio/8000/dataset981/{role}_{id}.wav'
        output_file_split = os.path.normpath(output_file).split(os.sep)
        output_directory = '/'.join(output_file_split[:-1])

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        synthesis_input = texttospeech.SynthesisInput(text=contents)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            # sample_rate_hertz = 44100
            sample_rate_hertz = 8000
        )
        
        requests = 5
        while requests:
            try:
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                audio_content = response.audio_content

                if os.path.exists(output_file):
                    output_audio = combinewav(output_file, audio_content)
                    output_audio.export(output_file, format='wav')
                    print(f'The audio content: {output_file} is overwritten.')
                    time.sleep(0.5)
                    break
                else:
                    with open(output_file, 'wb') as output:
                        output.write(audio_content)
                        print(f'The audio content is written to a file: {output_file}')
                        time.sleep(0.5)
                    break
                # save: bool = False
                # if save:
                #     with open(f'doc_output/{role}_{id}.wav', "wb") as output:
                #         output.write(response.audio_content)
                #         print(f'Audio content written to file: output/{role}_{id}.wav')
                
            except Exception as e:
                requests-=1
                if requests == 0:
                    log.logger.error(f'{e}, Failed to transcribe the audio: {output_file}')
                    

class SpeechToText():
    
    def __init__(self, credentials: str) -> str:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
        self.speechtotext_client = speech.SpeechClient()
        global logger
    
    def SpeechToText(self, speech_file: str) -> None:
        '''
        Transcribe the given audio file asynchronously.

        Note that transcription is limited to a 60 seconds audio file.
        
        Use a GCS file for audio longer than 1 minute.
        '''
        try:
            with open(speech_file, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                language_code="en-US",
            )

            requests = 5
            while requests:
                try:
                    operation = self.speechtotext_client.long_running_recognize(config=config, audio=audio)
                    
                except:
                    requests-=1
            response = operation.result(timeout=90)

            response_text = ' '.join(result.alternatives[0].transcript.strip() for result in response.results)
            return response_text
            
            # print(response.results)

            # for result in response.results:
            #     # The first alternative is the most likely one for this portion.
            #     print(f"Transcript: {result.alternatives[0].transcript}")
            #     print(f"Confidence: {result.alternatives[0].confidence}")
            
            # output_file_name = f"{output_directory.split('/')[-2]}.txt"
            # output_file = os.path.join(output_directory, output_file_name)
            # with open(output_file, 'w') as out:
            #     out.write(response.results)
        
        except Exception as e:
            print(e)


class Test():
    '''
    cloud log test
    '''
    def __init__(self) -> None:
        pass

    def test_log(self):
        log.logger.info('test')