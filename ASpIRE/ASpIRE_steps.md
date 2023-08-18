## STEP 1:

```bash
online2-wav-nnet3-latgen-faster   \
--online=false   \
--do-endpointing=false   \
--frame-subsampling-factor=3   
--config=exp/tdnn_7b_chain_online/conf/online.conf   \
--max-active=7000   \
--beam=15.0   \
--lattice-beam=6.0   \
--acoustic-scale=1.0   \
--word-symbol-table=exp/tdnn_7b_chain_online/graph_pp/words.txt   \
exp/tdnn_7b_chain_online/final.mdl   \
exp/tdnn_7b_chain_online/graph_pp/HCLG.fst   \
'ark:echo doctor_1 .|'   \
'scp:echo . doctor_1.wav |'   \
'ark,t:output_test.lat'
```

## STEP 2:

```bash
lattice-best-path --acoustic-scale=0.1 "ark,t:output_test.lat" "ark,t:output_test.txt"
```

## STEP 3:

```bash
utils/int2sym.pl -f 2- exp/tdnn_7b_chain_online/graph_pp/words.txt < output_test.txt > output_final.txt
```

## Creating the dictionary

First, we need to extract every unique word from the corpus, for that we can use this one liner:

```bash
grep -oE "[A-Za-z\\-\\']{3,}" corpus.txt | tr '[:lower:]' '[:upper:]' | sort | uniq > words.txt
```

Sequitur is an application that can be trained to generate readings for new words.

Build and install Sequitur G2P according to the instructions: 
- [https://www-i6.informatik.rwth-aachen.de/web/Software/g2p.html](https://www-i6.informatik.rwth-aachen.de/web/Software/g2p.html)
- [Build Status Sequitur G2P](https://github.com/sequitur-g2p/sequitur-g2p)

After installing, run the following script to train a model, as these will take quite a while, best leave it on for the weekend.

Run from ‘kaldi/egs/aspire/s5/data/local/dict/cmudict/sphinxdict’:
```console
g2p.py --train cmudict --devel 5% --write-model model-1
 
g2p.py --train cmudict_SPHINX_40 --devel 5% --write-model model-1
g2p.py --model model-1 --test cmudict_SPHINX_40 > model-1-test

g2p.py --model model-1 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-2
g2p.py --model model-2 --test cmudict_SPHINX_40 > model-2-test

g2p.py --model model-2 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-3
g2p.py --model model-3 --test cmudict_SPHINX_40 > model-3-test

g2p.py --model model-3 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-4
g2p.py --model model-4 --test cmudict_SPHINX_40 > model-4-test

g2p.py --model model-4 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-5
g2p.py --model model-5 --test cmudict_SPHINX_40 > model-5-test

g2p.py --model model-5 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-6
g2p.py --model model-6 --test cmudict_SPHINX_40 > model-6-test

g2p.py --model model-6 --ramp-up --train cmudict_SPHINX_40 --devel 5% --write-model model-7
g2p.py --model model-7 --test cmudict_SPHINX_40 > model-7-test
```

Finally, apply your generated model to your new words.

```
g2p.py --model model-7 --apply words.txt > words.dic
```

## Creating the language model

After creating your dictionary, it is also possible to build a language model using a tool called SRILM.

Convert the corpus to uppercase, because the dictionary is in uppercase format
```
cat corpus.txt | tr '[:lower:]' '[:upper:]' > corpus_upper.txt
```

## Execute SRILM:

Access this link to download related corpus and code. Copy 2 file corpus.txt and vocab.txt(vocabulary of words in corpus) into ‘\usr\share\srilm\bin\i686-gcc4’(for 32 bit) or “\usr\share\srilm\i686-m64”(for 64 bit). Manual copy might do not work, you could use Terminal, go to folder containing those files and type:

```console
$ sudo cp vocab.txt '/usr/share/srilm/bin/i686-m64'
$ sudo cp corpus.txt '/usr/share/srilm/bin/i686-m64'
```

Now move to above folder and run the program, generate a language model from the corpus
```console
$ cd '/usr/share/srilm/bin/i686-m64'
$ sudo ./ngram-count -text corpus_upper.txt -order 3 -limit-vocab -vocab words.txt -unk -map-unk "<unk>" -kndiscount -interpolate -lm lm.arpa
```

## Merging the input files

You might have to unzip the language model first:

```
gunzip -k data/local/lm/3gram-mincount/lm_unpruned.gz
```

To merge the dictionary and language model files we have generated with the ones used in the original model, you can use the script [mergedicts.py](https://github.com/MRAmateurH/code/blob/main/spider/meddialog/ASpIRE/mergedicts.py)

The next script assumes you are in the 'new' directory. If you have used the CMU web service, you will need the .dic and the .lm files.

```console
python3 mergedicts.py ../data/local/dict/lexicon4_extra.txt ../data/local/lm/3gram-mincount/lm_unpruned words.dic lm.arpa merged-lexicon.txt merged-lm.arpa
```

## Recompile the HCLG.fst

Now we have all the input files we need to re-compile the HCLG.fst using our new lexicon and grammar files.

Make sure you are in the ‘kaldi/egs/aspire/s5’ folder.

Let’s assume that you have created a ‘new/local/dict’ folder and copied over the following files from ‘data/local/dict’:
- extra_questions.txt
- nonsilence_phones.txt
- optional_silence.txt
- silence_phones.txt

you have copied your merged dictionary file into this folder as 'lexicon.txt'

you have created a new/local/lang folder with the merged language model in it called 'lm.arpa'

### Reference
[SRILM](https://hovinh.github.io/blog/2016-04-22-install-srilm-ubuntu/)
[Kaldi ASR: Extending the ASpIRE model](https://chrisearch.wordpress.com/2017/03/11/speech-recognition-using-kaldi-extending-and-using-the-aspire-model/)