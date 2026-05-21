Lo script principale è il file hm_script_1.m.
I vari passaggi sono spiegati nei commenti nel codice.

Per eseguire il programma, impostare il path dell'immagine da aprire e poi lanciare lo script.

Si apre una finestra con l'immagine, su cui bisogna individuare due coppie di rette parallele con direzione parallela al terreno; una retta è individuata selezionando due punti. 

A questo punto sono calcolati due punti di fuga, che consentono di individuare la retta di fuga delle direzioni parallele al terreno. Il programma va in pausa per mostrare questa retta di fuga, l'esecuzione riprende dando un qualsiasi input alla linea di comando matlab.

Ora si individua l'altezza di riferimento: si selezionano due punti che definiscono un segmento ortogonale al terreno (N.B. selezionare per primo il punto a terra); quindi si inserisce da linea di comando la lunghezza reale di questo segmento.

Per concludere, selezionare l'altezza da misurare; il segmento è definito selezionando i due estremi (N.B. selezionare per primo il punto a terra).


Possibili miglioramenti:
-Calcolare i punti di fuga come intersezione (ai minimi quadrati) di più rette, per aumentare la robustezza.
-Calcolare la retta di fuga indicando più punti per cui deve passare, per aumentare la robustezza.
-La rappresentazione della geometria in coordinate omogenee è usata solo da un certo punto in poi: sarebbe opportuno modificare il codice per usarla dall'inizio.
-Migliorare l'interfaccia. 

