from tkinter import Tk, ttk, StringVar, IntVar, constants
import sentenceAnalysis
import torchtext
import gensim.downloader
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from scipy.spatial.distance import cosine
import fasttext
import fasttext.util

class UI:
       def __init__(self, root):
              self._root = root
              self._label_var = None
              self._sentence_1 = None
              self._sentence_2 = None
              self._algo_opt = None
              self._result = None

       def start(self):            
              def run_test():
                     print("test")
                     
              def run_algo():
                     print(self._algo_opt.get())
                     if self._algo_opt.get() == "fastText":
                            self._result.set(str(sentenceAnalysis.get_ft_sim(self._sentence_1.get(), self._sentence_2.get(), ftmodel)))
                     elif self._algo_opt.get() == "word2vec":
                            self._result.set(str(sentenceAnalysis.get_w2v_sim(self._sentence_1.get(), self._sentence_2.get(), w2vmodel)))
                     elif self._algo_opt.get() == "GloVe":
                            self._result.set(str(sentenceAnalysis.get_glove_sim(self._sentence_1.get(), self._sentence_2.get(), glovemodel)))
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

              algo_opt_1.grid(row=11, column=0, padx=5, pady=5)
              algo_opt_2.grid(row=12, column=0, padx=5, pady=5)
              algo_opt_3.grid(row=13, column=0, padx=5, pady=5)

              run_button = ttk.Button(master=self._root, text="Run", command=run_algo)
              run_button.grid(row=15, column=0, padx=5, pady=5)

              button = ttk.Button(master=self._root, text="test", command=run_test)
              button.grid(row=16, column=0, padx=5, pady=5)

                            
              # Result
              self._result = StringVar()
              self._result.set("*similarity score comes here*")
              result_heading = ttk.Label(master=self._root, text="Similarity score:")
              result_heading.grid(row=20, column=0, padx=5, pady=5)

              result_score = ttk.Label(master=self._root, textvariable=self._result)
              result_score.grid(row=20, column=1, padx=5, pady=5)

window = Tk()
window.title("Interface test")

# models
ftmodel = fasttext.load_model('cc.en.300.bin')
w2vmodel = gensim.downloader.load('word2vec-google-news-300')
glovemodel = torchtext.vocab.GloVe(name='6B', dim=300)

ui = UI(window)
ui.start()

# Start loop
window.mainloop()