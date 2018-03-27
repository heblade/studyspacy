from __future__ import unicode_literals, print_function
import plac
import random
import spacy
import os

TRAIN_DATA = [
     ("Blade He is a developer, he is in this career over ten years."
      , {'entities': [(0, 8, 'PERSON')]}),
     ("Wendy is an employee in Morningstar"
      , {'entities': [(0, 5, "PERSON"),(24, 35,"ORG")]}),
     ("Google is a company to provide great products, Uber provides good car rent service."
      , {'entities': [(0, 6, "ORG"),(47, 51,"ORG")]})
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", str),
    n_iter=("Number of training iterations", "option", "n", int))
def train(model=None, output_dir=None, n_iter=100):
    try:
        nlp = spacy.load(model)
    except Exception as e:
        nlp = spacy.load("en_core_web_sm")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            count = itn + 1
            print("update %s times, losses: %s" % (count, losses))

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Test training Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Test training Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.to_disk(output_dir)
        print("Saved model to %s" % (output_dir))

        print("Loading from %s" % (output_dir))
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Test saved Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Test saved Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])



