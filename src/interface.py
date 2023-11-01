from tkinter import Tk, ttk, StringVar, IntVar, constants
import sentenceAnalysis
import torchtext
import gensim.downloader
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from scipy.spatial.distance import cosine
import fasttext
import fasttext.util
import datamuseSentence
import preprocessing
from sematch.semantic.similarity import YagoTypeSimilarity

class UI:
       def __init__(self, root):
              self._root = root
              self._label_var = None
              self._sentence_1 = None
              self._sentence_2 = None
              self._algo_opt = None
              self._result = None
              self._token1 = None
              self._token2 = None
              self._preprocess1 = None
              self._preprocess2 = None

       def start(self):            
                     
              def run_algo():
                     print(self._algo_opt.get())
                     self._token1.set(str(preprocessing.preprocess(self._sentence_1.get())))
                     self._token2.set(str(preprocessing.preprocess(self._sentence_2.get())))
                     self._preprocess1.set(str(preprocessing.preprocess2(self._sentence_1.get())))
                     self._preprocess2.set(str(preprocessing.preprocess2(self._sentence_2.get())))
                     if self._algo_opt.get() == "fastText":
                            self._result.set(str(sentenceAnalysis.get_ft_sim(self._preprocess1.get(), self._preprocess2.get(), ftmodel)))
                     elif self._algo_opt.get() == "word2vec":
                            self._result.set(str(sentenceAnalysis.get_w2v_sim(self._preprocess1.get(), self._preprocess2.get(), w2vmodel)))
                     elif self._algo_opt.get() == "GloVe":
                            self._result.set(str(sentenceAnalysis.get_glove_sim(self._preprocess1.get(), self._preprocess2.get(), glovemodel)))
                     elif self._algo_opt.get() == "Default":
                            self._result.set(str(datamuseSentence.tokensAll(self._sentence_1.get(), self._sentence_2.get())))
                     elif self._algo_opt.get() == "Best":
                            self._result.set(str(datamuseSentence.mode13(self._sentence_1.get(), self._sentence_2.get())))
                     elif self._algo_opt.get() == "YAGO":
                            try:
                                   msg = str(sentenceAnalysis.yagoSim(preprocessing.preprocess(self._sentence_1.get()), preprocessing.preprocess(self._sentence_2.get()), yago_sim))
                            except:
                                   msg = 'Sematch not functioning properly. See readme.'
                            self._result.set(msg)
                     elif self._algo_opt.get() == "BDpedia":
                            try:
                                   msg = str(sentenceAnalysis.bdpediaSim(preprocessing.preprocess(self._sentence_1.get()), preprocessing.preprocess(self._sentence_2.get()), concept))
                            except:
                                   msg = 'Sematch not functioning properly. See readme.'
                            self._result.set(msg)
                     else:
                            print("no vittu ei toimi vittu")
       
              
              # Sentence entries
              self._sentence_1 = StringVar()
              self._sentence_2 = StringVar()

              sent_heading = ttk.Label(master=self._root, text="Enter sentences")
              sent_heading.grid(row=0, column=0, padx=5, pady=5)

              sent_1_label = ttk.Label(master=self._root, text="Sentence 1")
              sent_1_label.grid(row=4, column=0, padx=5, pady=5)

              sent_1_entry = ttk.Entry(master=self._root, textvariable=self._sentence_1)
              sent_1_entry.grid(row=4, column=1, padx=5, pady=5)

              sent_2_label = ttk.Label(master=self._root, text="Sentence 2")
              sent_2_label.grid(row=4, column=2, padx=5, pady=5)

              sent_2_entry = ttk.Entry(master=self._root, textvariable=self._sentence_2)
              sent_2_entry.grid(row=4, column=3, padx=5, pady=5)


              #Algorithm options
              self._algo_opt = StringVar()

              algo_heading = ttk.Label(master=self._root, text="Select sentence processing algorithm")
              algo_heading.grid(row=10, column=0, padx=5, pady=5)

              algo_opt_1 = ttk.Radiobutton(master=self._root, text="fastText", variable=self._algo_opt, value="fastText")
              algo_opt_2 = ttk.Radiobutton(master=self._root, text="word2vec", variable=self._algo_opt, value="word2vec")
              algo_opt_3 = ttk.Radiobutton(master=self._root, text="GloVe", variable=self._algo_opt, value="GloVe")
              algo_opt_4 = ttk.Radiobutton(master=self._root, text="Datamuse default", variable=self._algo_opt, value="Default")
              algo_opt_5 = ttk.Radiobutton(master=self._root, text="Datamuse best", variable=self._algo_opt, value="Best")
              algo_opt_6 = ttk.Radiobutton(master=self._root, text="BDpedia concepts", variable=self._algo_opt, value="BDpedia")
              algo_opt_7 = ttk.Radiobutton(master=self._root, text="YAGO concepts", variable=self._algo_opt, value="YAGO")

              algo_opt_1.grid(row=11, column=0, padx=5, pady=5)
              algo_opt_2.grid(row=12, column=0, padx=5, pady=5)
              algo_opt_3.grid(row=13, column=0, padx=5, pady=5)
              algo_opt_4.grid(row=14, column=0, padx=5, pady=5)
              algo_opt_5.grid(row=11, column=1, padx=5, pady=5)
              algo_opt_6.grid(row=12, column=1, padx=5, pady=5)
              algo_opt_7.grid(row=13, column=1, padx=5, pady=5)

              run_button = ttk.Button(master=self._root, text="Run", command=run_algo)
              run_button.grid(row=16, column=0, padx=5, pady=5)

                            
              # Result
              self._result = StringVar()
              self._result.set("*similarity score comes here*")
              result_heading = ttk.Label(master=self._root, text="Similarity score:")
              result_heading.grid(row=23, column=0, padx=5, pady=5)

              result_score = ttk.Label(master=self._root, textvariable=self._result)
              result_score.grid(row=23, column=1, padx=5, pady=5)
              
              # tokens
              self._token1 = StringVar()
              self._token1.set("")
              result_heading = ttk.Label(master=self._root, text="Tokens:")
              result_heading.grid(row=21, column=0, padx=5, pady=5)

              result_score = ttk.Label(master=self._root, textvariable=self._token1)
              result_score.grid(row=21, column=1, padx=5, pady=5)
              
              self._token2 = StringVar()
              self._token2.set("")

              result_score = ttk.Label(master=self._root, textvariable=self._token2)
              result_score.grid(row=21, column=3, padx=5, pady=5)
              
              # preprocesses
              self._preprocess1 = StringVar()
              self._preprocess1.set("")
              result_heading = ttk.Label(master=self._root, text="Limited preprocess output:")
              result_heading.grid(row=20, column=0, padx=5, pady=5)

              result_score = ttk.Label(master=self._root, textvariable=self._preprocess1)
              result_score.grid(row=20, column=1, padx=5, pady=5)
              
              self._preprocess2 = StringVar()
              self._preprocess2.set("")

              result_score = ttk.Label(master=self._root, textvariable=self._preprocess2)
              result_score.grid(row=20, column=3, padx=5, pady=5)

window = Tk()
window.title("Interface test")

# models
ftmodel = fasttext.load_model('cc.en.300.bin')
w2vmodel = gensim.downloader.load('word2vec-google-news-300')
glovemodel = torchtext.vocab.GloVe(name='6B', dim=300)
yago_sim = YagoTypeSimilarity()
concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()),'models/dbpedia_type_ic.txt')

ui = UI(window)
ui.start()

# Start loop
window.mainloop()