import whisper


def speech_recognition(model='base'):
    """ Recognizes speech from an audio file passed to it """

    speech_model = whisper.load_model(model)
    result = speech_model.transcribe('YOUR_AUDIO_FILE', fp16=False)

    # write the recognized speech to a text file
    with open(f'transcription_{model}.txt', 'w') as file:
        file.write(result['text'])


def main():
    # models of whisper module (read Doc.)
    models = {1: 'tiny', 2: 'base', 3: 'small', 4: 'medium', 5: 'large'}

    for k, v in models.items():
        print(f'{k}: {v}')

    model = int(input('Choose a model by writing a number from 1 to 5: '))

    if model not in models.keys():
        raise KeyError(f'Model {model} is not in the list!')

    print('The transcription process has started, please wait...\n')
    speech_recognition(model=models[model])
    print('Готово!✅')


if __name__ == '__main__':
    main()
