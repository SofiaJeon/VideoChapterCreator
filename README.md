# 0. CodeFest 2020
해당 프로젝트는 [KC-ML2](https://www.kc-ml2.com/)에서 개최한 [CodeFest2020](https://blog.kc-ml2.com/codefest2020/)에 참가하여 진행한 내용입니다.

# 1. VideoChapterCreator
Automatic Video Chapter Creator with BERT NextSentencePrediction and KeySentenceGenerator

## 1.1 Introduction

Introducing **V**ideo **C**hapter creato**R**

Video Chapter creatoR(abbreviated **VCR**) is a technology that automatically creates chapters of videos. Videos, especially in the field of education, typically are long and tend to be one-shot-made. These videos are convenient to make in the aspect of video creator, however, not very efficient in the aspect of learner. 

There can be various purposes for learners to watch certain education video. For example, one might want to know every single details of the video. He or she can just watch the whole video and will be satisfied. 

*However*, other learners might not be sure whether he/she should watch this video or not, or might just want to skim through the video to search for few topics. For learners explained above, one-shot-made video, without any timeline that explains when this film deals with which topic, is inefficient. VCR is created, in order to solve this inefficiency.

## 1.2 Pipeline
![pipeline img](./img/pipeline_codefest.png)

## 1.3 Code Review
### 1.3.1 Package Install
```python
!pip install youtube_transcript_api
!pip install transformers

!python -m spacy download en
!git clone https://github.com/lovit/textrank
```

### 1.3.2 Transcripts, BERT NextSentencePrediction

```python
from youtube_transcript_api import YouTubeTranscriptApi
from urllib import parse
import re
import numpy as np
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

##transcripts_load
def transcripts_load(url):
        url_data = parse.urlparse(url)
        query = parse.parse_qs(url_data.query)
        video = query["v"][0]
        transcripts = YouTubeTranscriptApi.get_transcript(video)
        return transcripts

##transcripts split by seperator
def transcripts_split(transcripts, seperator):
    new_transcripts = []
    for transcript in transcripts:
        duration = 0
        if (transcript['text'][-1]!=seperator)&(seperator in transcript['text']):
            for text in transcript['text'].split(seperator):
                if transcript['text'].split(seperator).index(text)!=len(transcript['text'].split(seperator))-1:
                    new_transcript = {'text':text + seperator, 'start':transcript['start']+duration, 
                                    'duration':transcript['duration']/len(transcript['text'].split(seperator))}
                    new_transcripts.append(new_transcript)
                    duration = transcript['duration']/len(transcript['text'].split(seperator))
                else:
                    new_transcript = {'text':text, 'start':transcript['start']+duration, 
                                    'duration':transcript['duration']/len(transcript['text'].split(seperator))}            
                    new_transcripts.append(new_transcript)
        else:
            new_transcripts.append(transcript)            
    return new_transcripts
    
##transcripts sum
def transcripts_sum(transcripts):
    # 문장단위로 transcript 합치기
    new_transcripts = []
    temp_text = ''
    temp_start = 0
    temp_duration = 0

    for transcript in transcripts:
        if ('.'==transcript['text'][-1])|('?'==transcript['text'][-1])|('!'==transcript['text'][-1]):
            if temp_text:
                temp_text = temp_text + transcript['text'] + ' '
                temp_duration += transcript['duration']
                new_transcript = {'text':temp_text, 'start':temp_start, 'duration':temp_duration}

                new_transcripts.append(new_transcript)

                temp_text = ''
                temp_start = 0
                temp_duration = 0
            else:
                new_transcripts.append(transcript)
                temp_text = ''
                temp_start = 0
                temp_duration = 0
        else:
            temp_text = temp_text + transcript['text'] + ' '
            temp_duration += transcript['duration']
            temp_start = transcript['start']
            
    return new_transcripts

##transcripts remove stopwords
def transcripts_remove_stopwords(transcripts, stopwords):
    for transcript in transcripts:
        transcript['text'] = transcript['text'].lower()
        transcript['text'] = re.sub(r"\[([A-Za-z0-9_]+)\] ", '', transcript['text']).strip()
        transcript['text'] = re.sub(r"\[([A-Za-z0-9_]+)\]", '', transcript['text']).strip()
        transcript['text'] = transcript['text'].replace('\n', ' ').strip()
        for word in stopwords:
            transcript['text'] = transcript['text'].replace(word, '').strip()
            
    return transcripts

    
class Transcripts:
    def __init__(self, url, stopwords=['um, ', 'um,', 'um', 'uh, ', 'uh,', 'uh', 'you know, ', 'you know,']):
        self.transcripts = transcripts_load(url)
        self.stopwords = stopwords
        
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertNSP = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.losses = None
        self.threshold = None
        self.index = [-1]
        
    def transcripts_preprocess(self):
        # 문장단위로 transcript 나누기            
        new_transcripts = transcripts_split(self.transcripts, '.')
        new_transcripts = transcripts_split(new_transcripts, '?')
        new_transcripts = transcripts_split(new_transcripts, '!')
                
        # 문장단위로 transcript 합치기
        new_transcripts = transcripts_sum(new_transcripts)
        
        # 문장 내 의미없는 감탄사 등 지우기
        new_transcripts = transcripts_remove_stopwords(new_transcripts, self.stopwords)
        self.transcripts = new_transcripts
    
    def make_losses(self):
        self.losses = []

        for i in range(len(self.transcripts)-1):
            prompt = self.transcripts[i]['text']
            next_sentence = self.transcripts[i+1]['text']
            encoding = self.bertTokenizer(prompt, next_sentence, return_tensors='pt')

            loss, logits = self.bertNSP(**encoding, next_sentence_label=torch.LongTensor([1]))

            self.losses.append(float(loss))
            
    def make_threshold(self, percent=2):
        self.threshold = np.percentile(self.losses, percent)
        
    def transcripts_intergrate(self):
        for i in range(len(self.losses)):
            if self.losses[i] < self.threshold:
                self.index.append(i)
        
        transcripts_intergrated = []
        temp_text = ""
        temp_start = 0
        temp_duration = 0
        for i in range(len(self.transcripts)):
            if i-1 in self.index:
                if temp_text:
                    new_transcript = {'text':temp_text, 'start':temp_start, 'duration':temp_duration}
                    transcripts_intergrated.append(new_transcript)
                
                temp_text = ""
                temp_start = 0
                temp_duration = 0
                    
                temp_text += self.transcripts[i]['text']
                temp_start = self.transcripts[i]['start']
                temp_duration += self.transcripts[i]['duration']
            
            elif i==(len(self.transcripts)-1):
                temp_text += self.transcripts[i]['text']
                temp_duration += self.transcripts[i]['duration']
                new_transcript = {'text':temp_text, 'start':temp_start, 'duration':temp_duration}
                transcripts_intergrated.append(new_transcript)
            
            else:
                temp_text += self.transcripts[i]['text']
                temp_duration += self.transcripts[i]['duration']
        self.transcripts = transcripts_intergrated
```

```python
url = "https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5&ab_channel=StanfordUniversitySchoolofEngineering"
transcripts = Transcripts(url)
transcripts.transcripts_preprocess()
transcripts.make_losses()
transcripts.make_threshold()
transcripts.transcripts_intergrate()
```

### 1.3.3 Tokenizer

```python
import nltk, spacy
import gensim
from gensim.utils import simple_preprocess
nltk.download('stopwords')
# NLTK Stop words
from nltk.corpus import stopwords
```

```python
def manysents_to_words(sentences):
        return [simple_preprocess(str(sentence), deacc=True) for sentence in sentences]

class Tokenizer:
    def __init__(self, transcripts):
        self.paragraphs = [transcript['text'] for transcript in transcripts]
        self.start = [transcript['start'] for transcript in transcripts]
        self.stop_words = stopwords.words('english')
        self.words = manysents_to_words(self.paragraphs)
        
        # Build the bigram and trigram models
        self.bigram = gensim.models.Phrases(self.words, min_count=5, threshold=100) # higher threshold fewer phrases.
        self.trigram = gensim.models.Phrases(self.bigram[self.words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)

        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.sentences = None
    
    def sentence_tokenizer(self, sentence, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        #onesent_to_words
        tokens = simple_preprocess(sentence, deacc=True)

        #no_stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        #make_bigrams
        tokens = self.bigram_mod[tokens]

        #lemmatization
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    
    def make_sentences(self):
        self.sentences = []
        for paragraph in self.paragraphs:
            temp = [sentence.strip() for sentence in  paragraph.split('.') if sentence.strip()]
            self.sentences.append(temp)
```

```python
tokenizer = Tokenizer(transcripts.transcripts)
tokenizer.make_sentences()
```



### 1.3.4 Video Chapter creatoR

```python
from textrank.textrank.summarizer import KeysentenceSummarizer
from textrank.textrank.summarizer import KeywordSummarizer
```

```python
import datetime

summarizer = KeysentenceSummarizer(
    tokenize = tokenizer.sentence_tokenizer,
    min_sim = 0.3,
    verbose = False
)

for i in range(len(tokenizer.sentences)):
    #if len(tokenizer.sentences[i])>5:
    keysents = summarizer.summarize(tokenizer.sentences[i], topk=1)
    start = tokenizer.start[i]
    print(str(datetime.timedelta(seconds=int(start))), end=' ')
    for _, _, sent in keysents:
        print(sent, end=' ')
    print()
    print()
```

### 1.3.5 Sample

0:00:08 okay, so what we just saw earlier, this is taking one filter, sliding it over all of the spatial locations in the image and then we're going to get this activation map out, right, which is the value of that filter at every spatial location 

0:25:39 right, and so just kind of looking at the way that we computed how many, what the output size is going to be, this actually can work into a nice formula where we take our dimension of our input n, we have our filter size f, we have our stride at which we're sliding along, and our final output size, the spatial dimension of each output size is going to be n minus f divided by the stride plus one, right, and you can kind of see this as a, if i'm going to take my filter, let's say i fill it in at the very last possible position that it can be in and then take all the pixels before that, how many instances of moving by this stride can i fit in 

0:36:49 so remember each filter is going to do a dot product through the entire depth of your input vole 

0:38:46 [muffled speaking] yeah, so the question is, does the zero padding add some sort of extraneous features at the corners?and yeah, so i mean, we're doing our best to still, get some value and do, like, process that region of the image, and so zero padding is kind of one way to do this, where i guess we can, we are detecting part of this template in this region 

0:39:36 okay, so, yeah, so we saw how padding can basically help you maintain the size of the output that you want, as well as apply your filter at these, like, corner regions and edge regions 

0:44:19 can you guys speak up?250, okay so i heard 250, which is close, but remember that we're also, our input vole, each of these filters goes through by depth 


### 1.4 Limitation and Improvement

Currently, VCR only works on English transcripts. This technology does not support other languages yet. More and more languages will be supported soon. 

Also, we only support videos uploaded on Youtube, so videos should first be uploaded on Youtube before using this technology. 

We hope that by fine tuning on BERT, the quality of chunks - by quality, it means chunks that contain exactly one topic - can be improved. Proper evaluation method for the quality of the chunk is also needed. To mention about Key Sentence Generator, if there is better model to use instead, topic sentences can be more orderly, with proper length.


