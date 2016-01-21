from collections import defaultdict
import numpy as np


class Tagger():
    def __init__(self) :
        # log probabilités de transitions initiales P( . | début de phrase)
        # Par exemple, self.initial_transitions["NOUN"] doit contenir log P(NOUN | début de phrase)
        self.initial_transitions    = defaultdict(float)

        # log probabilités de transitions finales   P(fin de phrase | . )
        # Par exemple, self.final_transitions["NOUN"] doit contenir log P(fin de phrase | NOUN)
        self.final_transitions      = defaultdict(float)

        # log probabilités de transitions P(. | .)
        # Par exemple, self.transitions["NOUN"]["DET"] doit contenir log P(NOUN | DET)
        self.transitions            = defaultdict(lambda : defaultdict(float))

        # log probabilités d'émissions
        # Par exemple, self.emissions["chat"]["NOUN"] doit contenir log P(chat | NOUN)
        self.emissions              = defaultdict(lambda : defaultdict(float))

        self.tags = []              # liste (dédoublonnée) des tags possibles

    def train(self, sentences_lst, smooth=1e-5):
        """
        Estime les paramètres du modèle à partir d'un corpus d'entraînement.

        paramètres
        ----------
            sentences_lst : liste dont chaque élément est un couple de listes comme : (["le", "chat", "dort"], ["DET", "NOUN", "VERB"])
            smooth : valeur pour le lissage

        TODO (question2) : implémentez cette fonction
            - ne pas oublier d'effectuer un lissage
            - ne pas oublier de transformer les probabilités en log probabilités à la fin de la fonction
        """

        # comptage
        tag_freqs = defaultdict(float)                  # comptes de chaque catégorie
        for x,y in sentences_lst :
            self.initial_transitions[y[0]] += 1
            self.emissions[x[0]][y[0]] += 1
            tag_freqs[y[0]] += 1
            for i in range(1, len(x)) :
                self.emissions[x[i]][y[i]] += 1
                self.transitions[y[i]][y[i-1]] += 1
                tag_freqs[y[i]] += 1
            self.final_transitions[y[-1]] += 1

        self.tags = sorted(self.transitions)            # on récupère la liste des tags possibles


        # normalisation des transitions
        for tag in self.tags :
            self.initial_transitions[tag] = np.log((self.initial_transitions[tag] + smooth) / len(sentences_lst))     # P(tag | début de phrase) = compte(début, tag) / compte(début de phrase)
            for prev_tag in self.tags :
                self.transitions[tag][prev_tag] = np.log((self.transitions[tag][prev_tag] + smooth) / tag_freqs[prev_tag])   # P(t2 | t1) = compte(t1,t2) / compte(t1)
            self.final_transitions[tag] = np.log((self.final_transitions[tag] + smooth) / tag_freqs[tag])                    # P(fin | t) = compte(t, fin) / compte(t)

        # normalisation des émissions
        for word in self.emissions :
            for tag in self.tags :
                 self.emissions[word][tag] = np.log((self.emissions[word][tag] + smooth) / tag_freqs[tag]) # P(word | tag) = compte(word, tag) / compte(tag)


    def predict(self, sentence):
        """
        Prédit une séquence de tags à partir de la phrase

        sentence : liste de mots
        """
        emissions_scores = [self.emissions[word] for word in sentence]
        y = self.viterbi(self.initial_transitions, self.final_transitions, self.transitions, emissions_scores)
        return y

    def test(self, sentences_lst) :
        """
        Évalue le modèle sur un corpus (sentences_lst)

        Renvoie l'exactitude du modèle.
        """
        pred_count = defaultdict(float)
        tag_count = defaultdict(float)

        acc = 0.0
        tot = 0.0
        for x,y in sentences_lst :
            y_hat = self.predict(x)
            tot += len(y_hat)
            for i,tag in enumerate(y_hat) :
                pred_count[(tag, y[i])] += 1
                tag_count[tag] += 1

                if tag == y[i] :
                    acc += 1

        confusion = {}
        for (tag, pred), count in pred_count.items():
            confusion[(tag, pred)] = count / tag_count[tag]

        return (acc / tot, confusion)

    def viterbi(self, initial_transitions, final_transitions, transitions, emissions) :
        """
        Prédit la meilleure séquence de tags en utilisant l'algorithme de Viterbi

        Paramètres
        ----------

            initial_transitions : dictionnaire
                initial_transitions[tag] contient log(P(tag | début de phrase))
            final_transitions : dictionnaire
                final_transitions[tag] contient log(P(fin de phrase | tag))
            transitions : dictionnaire de dictionnaire
                transitions[tag][prev_tag] contient log(P(tag | prev_tag))
            emissions : liste de dictionnaire
                emissions[i][tag] contient log(P(w_i | tag))    (w_i est le ième mot de la phrase)

        TODO (question1) : finir d'implémenter cette fonction
        """


        n_classes = len(self.tags)   # n_classes est le nombre de tags différents
        n_words = len(emissions)     # n_words est la longueur de la phrase

        # scores devra contenir les poids de chemins dans le graphe
        scores = np.zeros((n_classes, n_words), dtype = float) - np.inf
        # backtrack sert à stocker les chemins
        backtrack = np.zeros((n_classes, n_words), dtype = int) - 1


        # scores initiaux
        for j,tag in enumerate(self.tags) :
            scores[j,0] =  emissions[0][tag] + initial_transitions[tag]

        # pour chaque mot de la phrase
        for i in range(1, n_words) :
            # pour chaque tag possible
            for j,tag in enumerate(self.tags) :

                # scorage de toutes les arêtes entrantes
                scores_tag = [emissions[i][tag] + scores[iprev][i-1] + transitions[tag][prev_tag] for iprev,prev_tag in enumerate(self.tags)]
                # détermination du meilleur prédécesseur
                best_idx,best_score = max(enumerate(scores_tag), key = lambda x : x[1])
                # mise à jour du score de l'état
                scores[j,i] = best_score
                # lien vers son prédecesseur
                backtrack[j,i] = best_idx

        for j,tag in enumerate(self.tags) :
            scores[j,-1] += final_transitions[tag]

        # détermination du meilleur puis de la meilleure séquence de tags
        sequence = np.zeros(n_words, dtype=int) -1
        sequence[-1] = np.argmax(scores[:,-1])
        for i in reversed(range(n_words-1)) :
            sequence[i] = backtrack[sequence[i+1], i+1]
        return [self.tags[i] for i in sequence]



#START = "<start>"
#STOP = "ROOT"

#class Tagger:
    #def __init__(self):
        #pass

    #def train(self, corpus, smooth=1e-5):
        #tag_count = defaultdict(float)
        #prev_tag_count = defaultdict(float)
        #emissions = defaultdict(float)
        #transitions = defaultdict(float)

#class Tagger:
    #def __init__(self):
        #pass

    #def train(self, corpus, smooth=1e-5):
        #tag_count = defaultdict(float)
        #prev_tag_count = defaultdict(float)
        #emissions = defaultdict(float)
        #transitions = defaultdict(float)

        #for sent in corpus:
            #words = sent[0]
            #cpos = sent[1]['cpos']

            #for i, w in enumerate(words[1:]):
                ##print("%i, %s" % (i, w))
                #prev_tag = cpos[i]
                #tag = cpos[i+1]
                ##print("\tprev: %s" % prev_cat)
                ##print("\tcat : %s" % cat)

                ##print(w)

                #tag_count[tag] += 1
                #prev_tag_count[prev_tag] += 1
                #emissions[(w, tag)] += 1
                #transitions[(prev_tag, tag)] += 1


            #print(tag_count)
            #print(prev_tag_count)
            #print(emissions)
            #print(transitions)

        #for sent in corpus:
            #words = sent[0]
            #cpos = sent[1]['cpos']

            #for i, w in enumerate(words[1:]):
                ##print("%i, %s" % (i, w))
                #prev_tag = cpos[i]
                #tag = cpos[i+1]
                ##print("\tprev: %s" % prev_cat)
                ##print("\tcat : %s" % cat)

                ##print(w)

                #tag_count[tag] += 1
                #prev_tag_count[prev_tag] += 1
                #emissions[(w, tag)] += 1
                #transitions[(prev_tag, tag)] += 1


            #print(tag_count)
            #print(prev_tag_count)
            #print(emissions)
            #print(transitions)

