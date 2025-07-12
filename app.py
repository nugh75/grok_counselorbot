from flask import Flask, render_template, request, jsonify, session
import ollama
import json
import logging
import datetime
import os
import requests
from flask_session import Session  # Per session server-side
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output su console
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'super_secret_key')  # Usa variabile d'ambiente
app.config['SESSION_TYPE'] = 'filesystem'  # Per persistenza session
Session(app)

# Configurazione Ollama remota
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://192.168.129.14:11435')
logger.info(f"🔗 Configurazione Ollama: {OLLAMA_HOST}")
client = ollama.Client(host=OLLAMA_HOST)

# Configurazione OpenRouter
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

# Lista di modelli disponibili su Ollama
OLLAMA_MODELS = [
    'mixtral:8x7b',
    'llama4:16x17b', 
    'qwen2.5:7b',
    'qwen3:32b',
    'hf.co/Mungert/MiMo-VL-7B-RL-GGUF:BF16',
    'devstral:24b',
    'phi4:14b',
    'deepseek-r1:latest',
    'qwen:110b',
    'llama3:latest'
]

# Lista di modelli gratuiti OpenRouter
OPENROUTER_FREE_MODELS = [
    'microsoft/phi-3-mini-128k-instruct:free',
    'microsoft/phi-3-medium-128k-instruct:free',
    'mistralai/mistral-7b-instruct:free',
    'huggingfaceh4/zephyr-7b-beta:free',
    'openchat/openchat-7b:free',
    'gryphe/mythomist-7b:free',
    'undi95/toppy-m-7b:free',
    'openrouter/cinematika-7b:free',
    'google/gemma-7b-it:free',
    'meta-llama/llama-3-8b-instruct:free'
]

# Combina tutti i modelli disponibili
MODELS = OLLAMA_MODELS + OPENROUTER_FREE_MODELS

# Modelli che supportano meglio i tool calls
TOOL_COMPATIBLE_MODELS = [
    'mixtral:8x7b',
    'qwen2.5:7b', 
    'qwen3:32b',
    'phi4:14b',
    'llama3:latest',
    # Modelli OpenRouter compatibili
    'microsoft/phi-3-medium-128k-instruct:free',
    'mistralai/mistral-7b-instruct:free',
    'meta-llama/llama-3-8b-instruct:free'
]

# Prompt di sistema (come prima)
SYSTEM_PROMPT = """
Personalità

Sei un compagno di apprendimento amichevole e disponibile, di nome counselorbot . Aiuti gli utenti a migliorare le proprie strategie di apprendimento. Sei paziente, incoraggiante e fornisci feedback costruttivi.
Ambiente

Stai interagendo con un utente che ha appena completato il "Questionario Strategie di Apprendimento" (QSA) sul sito competenzestrategiche.it. Il QSA è un questionario di self-assessment, ovvero di autovalutazione, che aiuta l’utente a riflettere sulle proprie abitudini e strategie nello studio. L'utente cerca feedback e approfondimenti sui propri risultati. Hai accesso a informazioni generali sulle strategie di apprendimento, ma non puoi accedere direttamente ai risultati specifici dell'utente. Presumi che l’utente sia un adulto interessato al miglioramento personale.
Tono

Le tue risposte sono positive, incoraggianti e di supporto. Usi un linguaggio chiaro e semplice, evitando il gergo tecnico. Sei conversazionale e coinvolgente, usando frasi come “Che interessante!” o “Parlami di più di...”. Sei paziente e comprensivo, e lasci spazio all’utente per esprimere pensieri ed emozioni.
Obiettivo

Il tuo obiettivo principale è aiutare l’utente a comprendere i risultati del QSA e a identificare aree di miglioramento nelle sue strategie di apprendimento. Segui questi passaggi:
Procedura operativa

    Richiesta dei dati iniziali
    Sollecitare l’utente a fornire i punteggi dei fattori cognitivi del QSA (C1 – C7) e di quelli affettivo-motivazionali (A1 – A7).

    Analisi di primo livello
        Registrazione dei valori ricevuti tramite lo strumento risultati.
        Esame di ciascun fattore con il tool af-qsa, illustrandone la portata operativa e proponendo spunti di riflessione mirati.

    Proposta di approfondimento
    Dopo l’analisi di primo livello, suggerire all’utente un’ulteriore indagine di secondo livello.

    Analisi di secondo livello (se richiesta)
        Riorganizzare i dati secondo i raggruppamenti tematici e analizzarli in modo trasversale con il tool sl-qsa.
        Evidenziare connessioni, punti di forza e criticità emergenti.

Regole

Parla sempre in italiano.
Evita di dare consigli specifici o interpretazioni dei risultati individuali del QSA, poiché non ne hai accesso. Non fornire consigli medici o psicologici. Non chiedere informazioni personali identificabili (PII). Mantieni il focus su strategie di apprendimento generali e su risorse utili. Se l’utente esprime frustrazione o confusione, offri rassicurazione e suggerisci di suddividere il compito in passaggi più piccoli.
Tools

af-qsa usa questo tool per l'analisi fattore per fattore sl-qsa usa questo tool per l'analisi di secondo livello d-qsa usa questo tool per una analisi approfondite

IMPORTANTE: Quando ricevi punteggi QSA dall'utente, devi SEMPRE utilizzare prima il tool "risultati" per registrarli, poi il tool "af-qsa" per l'analisi. Non scrivere mai la sintassi dei tool calls come testo normale, ma usali direttamente.
"""

# Dati per i tool (testi completi estratti dal messaggio fornito)

AF_QSA_DATA = {
    "C1": """**C1 strategie elaborative:** Il fattore si riferisce alla conoscenza, sensibilità e capacità di usare strategie di natura elaborativa al fine di comprendere e ricordare quanto studiato.
Un basso punteggio in questo fattore denota difficoltà a gestire bene lo studio e i processi di memorizzazione e la tendenza a studiare in modo disorganico, senza mettere in relazione quanto studiato con quanto conosciuto, con la propria esperienza, con immagini mentali o con esempi. È dunque opportuno quando si studia: cercare esempi; applicare quanto si studia a situazioni personali; collegare i vari passaggi a immagini significative; ripetere mentalmente in modo sistematico; usare analogie; trarre conclusioni non esplicitate nel testo; selezionare progressivamente gli elementi fondamentali del discorso e relazionarli tra loro; avere chiari gli obiettivi che si vogliono raggiungere.""",

    "C2": """**C2 autoregolazione:** Il fattore si riferisce alla capacità di gestione autonoma dello studio e dei processi di apprendimento, alla accuratezza e riflessività nello svolgere i propri impegni di studio.
Un basso punteggio in questo fattore denota la tendenza a studiare con scarsa attenzione, la incapacità a gestire lo studio in maniera autonoma e la mancanza di un metodo preciso.
È dunque opportuno cercare di tenere sotto controllo le proprie azioni: all’inizio dello studio verificare quali sono le cose da fare; prendere appunti durante le lezioni e rivederli dopo le lezioni, segnare sul testo le cose più importanti; organizzare lo studio in base al tempo a disposizione.""",

    "C3": """**C3 Disorientamento** Il fattore tocca due aspetti fondamentali delle strategie di studio: la capacità o meno di portare a termine gli impegni e quella di organizzare le conoscenze studiate in modo da conservarle meglio nel tempo.
Un alto punteggio in questo fattore è collegabile a una scarsa capacità di organizzare in maniera produttiva il materiale da studiare, il tempo a disposizione e l’ambiente in vista dell’obiettivo da raggiungere.
È dunque opportuno imparare ad organizzare positivamente il materiale da studiare, il tempo a disposizione, l’ambiente di studio.""",

    "C4": """**C4 Disponibilità alla collaborazione:** Il fattore mira ad identificare l’apprezzamento e la disponibilità ad uno studio partecipativo e collaborativo sia al fine di migliorare il proprio apprendimento, sia a quello di imparare a lavorare in gruppo.
Punteggi bassi in questo fattore sono collegabili a una scarsa disponibilità alla collaborazione in gruppo.
È dunque opportuno considerare l’importanza e l’utilità di uno studio partecipativo e collaborativo al fine di migliorare il proprio apprendimento ed imparare a lavorare in un gruppo.""",

    "C5": """**C5 organizzatori semantici:** Il fattore mira ad identificare la capacità di utilizzazione di organizzatori semantici (diagrammi, schemi, mappe...).
Un punteggio basso in questo fattore segnala una debole utilizzazione di organizzatori semantici, strumenti di lavoro intellettuale particolarmente importanti non solo per comprendere ma anche per organizzare in maniera coerente e sistematica quello che si studia o si ascolta, per facilitare il ricordo e per favorire la capacità di risolvere problemi.
È dunque opportuno imparare ad utilizzare alcuni importanti organizzatori semantici come schemi, tabelle, diagrammi, mappe concettuali.""",

    "C6": """**C6 Difficoltà di concentrazione: **Il fattore mira ad identificare la capacità di gestire il proprio ambiente e il proprio tempo per garantire uno spazio adeguato alla concentrazione nel lavoro.
Alti punteggi in questo fattore denotano un soggetto che si distrae facilmente oppure non è in grado di focalizzare l’attenzione per un tempo adeguato allo svolgimento di un compito.
È dunque opportuno sviluppare valide ed efficaci strategie di controllo dell’attenzione: evitare un’eccessiva esposizione alla televisione; eliminare le fonti di distrazione; leggere in maniera sistematica; porre e porsi spesso domande; conoscere gli obiettivi che si vogliono raggiungere con le varie attività didattiche.""",

    "C7": """**C7 Autointerogazione:** Il fattore si riferisce alla tendenza a porsi domande o al porre domande agli altri come strategie di controllo della comprensione, del ricordo o dello stato di preparazione per verificare il livello di apprendimento raggiunto.
Bassi punteggi in questo fattore indicano una difficoltà ad usare queste strategie per controllare i risultati del proprio studio.
È dunque opportuno: porsi domande e porre domande agli altri, all’insegnante, ai compagni; usare le domande inserite nel testo per capire e ricordare; prefigurare le domande che potranno essere fatte in sede d’esame.""",

    "A1": """**A1 ansietà di base:** Un punteggio elevato in questo fattore denota la tendenza a vivere il percorso di studio in modo essenzialmente emotivo. L’emozione è una componente molto importante e, quando sia presente, potenzia notevolmente l’elemento cognitivo nel percorso di apprendimento. Naturalmente la componente emotiva potrà generare un’ansia moderata o, al contrario, un’ansia eccessiva. Va da sé che un’ansia moderata consente di vivere il proprio percorso di apprendimento e le proprie prestazioni in modo positivo e stimolante, mentre un’ansia eccessiva può diventare paralizzante e condurre ad errori o all’inazione. È dunque opportuno che lo studente, anche opportunamente indirizzato, impari a riflettere sugli eventuali elementi che possano provocargli un’ansia eccessiva per ricondurli ad una dimensione accettabile e gestibile della stessa, in modo da renderla funzionale alle proprie aspettative di conoscenza e di performance.""",

    "A2": """**A2 Volizione:** La motivazione è la condizione interna che produce energia atta a dirigere il comportamento verso una meta ben precisa. Può accadere che eventi interni o esterni, provochino una caduta della motivazione tale da distogliere il soggetto dalla tensione positiva verso il raggiungimento dello scopo prefisso. È importante che lo studente sappia riconoscere questo particolare stato emotivo, per poterne analizzare le cause e per poter dunque, opportunamente indirizzato, tentare una analisi ed una relativizzazione ed eventuale rimozione delle stesse.""",

    "A3": """**A3 Attribuzione a cause controllabili:** La capacità di attribuire i propri successi o fallimenti a cause controllabili, determina una chiara percezione dei fattori, interni o esterni, che possano favorire o ritardare il proprio apprendimento. Al contrario, un basso punteggio in questo fattore denota una tendenza ad attribuire il proprio successo o fallimento ad elementi che si ritengono fuori dalla sfera di influenza del soggetto stesso. È importante che lo studente acquisisca la consapevolezza che l’intelligenza e la capacità di apprendere non sono definiti una volta per tutte, ma che sono fattori in evoluzione e che la continua modificazione degli stessi può essere operata dallo stesso soggetto in apprendimento, con il necessario impegno ed opportunamente guidato.""",

    "A4": """**A4 Attribuzione a cause incontrollabili:** Un alto punteggio in questo fattore denota una tendenza ad attribuire il proprio successo o fallimento ad elementi che si ritengono fuori dalla sfera di influenza del soggetto stesso.
Dunque, lo studente che risponde a questo profilo, mostrerà una debolezza nella capacità di attivare un reale controllo del proprio processo di apprendimento, attribuendo i risultati dello stesso a cause che non dipendono dalla propria abilità nel gestirlo. Sarà perciò importante attivare percorsi di riflessione e di analisi sulle cause, in modo da definirne i confini di realtà e procedere pertanto ad una attivazione della metaconoscenza dei propri personali tempi e modi nell’apprendere. È importante che lo studente acquisisca la consapevolezza che l’intelligenza e la capacità di apprendere non sono definiti una volta per tutte, ma che sono fattori in evoluzione e che la continua modificazione degli stessi può essere operata dallo stesso soggetto in apprendimento, opportunamente guidato.""",

    "A5": """**A5 Mancanza di perseveranza:** Il fattore si riferisce alla capacità di perseverare per portare a termine un impegno o un compito assegnato.
Un alto punteggio mette in luce la fragilità della capacità di perseverare nel lavoro e nel portare a termine un impegno di studio. Come ci si può attendere, il fattore è anche correlato negativamente al fattore A2 (Volizione). Ed è inoltre correlato positivamente con il terzo fattore cognitivo (Disorientamento). Soggetti che ottengono punteggi elevati in questo fattore (A5, mancanza di perseveranza) e nel fattore disorientamento, mentre conseguono un punteggio basso nel fattore volizione, tendono anche ad attribuire i loro risultati a cause incontrollabili. Si tratta certamente di soggetti da tenere sotto osservazione. È evidente la necessità di impostare programmi di insegnamento esplicito di strategie di studio sia di natura cognitiva, sia motivazionale.""",

    "A6": """**A6 Percezione di competenza:** La percezione della propria competenza nel portare a termine gli impegni scolastici, è considerata dagli esperti uno dei fattori principali nello sviluppo di motivazioni e disposizioni positive nel percorso di apprendimento. Il senso di responsabilità, a sua volta, sostiene e potenzia la dimensione positiva della correlazione tra compito da eseguire, soddisfazione per il lavoro fatto e stima di sé. Può accadere che nel percorso di apprendimento si venga deviati dal circolo positivo: responsabilità verso il compito – soddisfazione per l’esecuzione del compito – stima di sé – percezione di competenza – potenziamento della consapevolezza delle proprie capacità nell’apprendimento – rinnovata responsabilità verso un nuovo compito più elevato – e così via. Qualora questo accada è importante analizzare l’anello debole, o l’interruzione della catena positiva per ripristinarne la continuità e ritrovare il ritmo giusto nella relazione tra il sé e il compito da portare a termine.""",

    "A7": """**A7 Interferenze emotive:** L’emozione è una componente molto importante e, quando sia presente, potenzia notevolmente l’elemento cognitivo nel percorso di apprendimento. Naturalmente la componente emotiva potrà generare un’ansia moderata o, al contrario, un’ansia eccessiva. Va da sé che un’ansia moderata consente di vivere il proprio percorso di apprendimento e le proprie prestazioni in modo positivo e stimolante, mentre un’ansia eccessiva può diventare paralizzante e condurre ad errori o all’inazione. È dunque opportuno che lo studente, anche opportunamente indirizzato, impari a riflettere sugli eventuali elementi che possano provocargli un’ansia eccessiva per ricondurli ad una dimensione accettabile e gestibile della stessa, in modo da renderla funzionale alle proprie aspettative di conoscenza e di performance"""
}

SL_QSA_DATA = {
    "Strategie elaborative per comprendere e studiare": """**Strategie elaborative per comprendere e studiare**: C1, C5, C7. Questi fattori insieme indicano la capacità di ricordare, memorizzare e connettere quanto si è studiato alle proprie conoscenze pregresse. Le strategie elaborative permettono di creare connessioni significative tra le informazioni nuove e quelle già acquisite, favorendo una comprensione profonda e duratura. È importante utilizzare tecniche come la creazione di mappe concettuali, l'uso di analogie e la sintesi delle informazioni per migliorare l'apprendimento.""",

    "Autoregolazione e pianificazione": """**Autoregolazione e pianificazione**: C2, C3, C6. Questi fattori insieme indicano le capacità di pianificare ed autoregolarsi. La pianificazione consente di organizzare il tempo e le risorse in modo efficace, mentre l'autoregolazione aiuta a monitorare e adattare il proprio approccio allo studio. Questi fattori sono strettamente legati alle strategie elaborative e ai fattori cognitivi, poiché una buona pianificazione e autoregolazione si basano su una comprensione chiara degli obiettivi e delle priorità. È utile sviluppare un metodo di studio strutturato, che includa la definizione di obiettivi specifici, la gestione delle distrazioni e la revisione periodica dei progressi.""",

    "Disponibilità alla collaborazione": """**Disponibilità alla collaborazione**: C4. Questo fattore ci dice quanto siamo inclini a prediligere lo studio con altri o individuale. La collaborazione può arricchire l'apprendimento attraverso lo scambio di idee e il confronto con prospettive diverse. Tuttavia, è importante bilanciare il lavoro di gruppo con momenti di studio individuale per consolidare le conoscenze acquisite. La disponibilità alla collaborazione favorisce anche lo sviluppo di competenze sociali e la capacità di lavorare in team, che sono essenziali in molti contesti professionali.""",

    "Controllo delle proprie emozioni e dell'ansia": """**Controllo delle proprie emozioni e dell'ansia**: A1, A7. Questi fattori da una parte sono legati allo stress, all'arousal e alle interferenze emotive che possono, se troppo alti, bloccare e far prendere scelte d'impulso allo studente. Dall'altra, se moderati, possono essere un aiuto alla concentrazione e al mantenimento della tensione. Un po' di ansia prima di una prova importante ci aiuta a essere vigili e concentrati, ma è fondamentale imparare a gestire le emozioni per evitare che diventino paralizzanti. Tecniche di rilassamento, come la respirazione profonda e la meditazione, possono essere utili per mantenere un equilibrio emotivo.""",

    "Capacità di impegnarsi e di essere concentrati sui propri obiettivi": """**Capacità di impegnarsi e di essere concentrati sui propri obiettivi**: A2, A5. Questi fattori sono simmetrici e ci permettono di impegnarci in quello che facciamo, di essere perseveranti, non perdere il focus, gestire le priorità e andare verso il nostro obiettivo. La capacità di mantenere la concentrazione e la determinazione è essenziale per superare le difficoltà e raggiungere i propri obiettivi. È importante sviluppare una mentalità orientata al successo, che includa la definizione di obiettivi chiari, la gestione del tempo e la capacità di affrontare le sfide con resilienza.""",

    "Locus of control e percezione di competenza": """**Locus of control e percezione di competenza**: A3,A4,A6.Il locus of control rappresenta gli stili attributivi, ovvero la tendenza ad attribuire i propri successi o insuccessi a cause controllabili interne o esterne. È collegato alla percezione di competenze. Una persona che si sente competente in una certa materia o compito sarà più propensa ad avere una maggiore agency. La percezione di competenza influisce direttamente sulla motivazione e sulla fiducia in sé stessi. È importante aiutare gli studenti a sviluppare una visione positiva delle proprie capacità, incoraggiandoli a riconoscere i propri punti di forza e a lavorare sulle aree di miglioramento."""
}

D_QSA_DATA = {
    "C1": """C1 Strategie elaborative
Domanda: Sai mettere in relazione i nuovi concetti con quanto già conosci, con la tua esperienza o con immagini mentali allo scopo di comprendere e ricordare meglio?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Cerco di trovare legami tra ciò che sto studiando e le mie esperienze (22)
Cerco di trovare le relazioni tra ciò che apprendo e ciò che già conosco (17)
Durante lo studio e l'ascolto di una lezione mi vengono in mente collegamenti con altri argomenti già studiati (48)
Cerco di stabilire collegamenti tra le diverse idee esposte nel testo che studio (31)
Per ricordare meglio quanto studio, cerco di collegare tra loro le varie idee (100)
Quando imparo un nuovo concetto cerco di trovare un esempio a cui esso si possa applicare (36)
Quando imparo nuove parole o nuove idee cerco di immaginare una situazione a cui esse si possano applicare (7)
Quando imparo un nuovo concetto mi domando se ci sono casi in cui esso non può essere applicato (41)
Cerco di vedere come ciò che studio potrebbe applicarsi alla mia vita di tutti i giorni (26)
Leggendo ricostruisco con la mia immaginazione le situazioni, i personaggi o le vicende narrate (85)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Quando studi cerca esempi
Applica i nuovi concetti a situazioni personali
Collega i vari passaggi e i concetti a immagini significative
Ripeti mentalmente
Usa analogie
Cerca di trarre conclusioni non esplicitate nel testo
Seleziona progressivamente gli elementi fondamentali del discorso e collegali tra loro
Chiarisci gli obiettivi che vuoi raggiungere""",

    "C5": """C5 Uso di organizzatori semantici
Domanda: Sai organizzare in modo coerente e sistematica quello che studi? Ti servi di schemi, disegni, grafici e tabelle per comprendere e ricordare meglio?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Gli schemi, i grafici o le tabelle riassuntive mi aiutano a capire meglio quanto esposto nel testo (18)
Capisco meglio se l'insegnante nello spiegare usa schizzi e grafici fatti sulla lavagna (90)
Faccio disegni e schizzi che mi aiutano a comprendere quello che sto studiando (44)
Mi costruisco schemi, grafici o tabelle riassuntive per sintetizzare ciò che studio (37)
Ricordo meglio quando studio se posso servirmi di schemi, grafici o tabelle (56)
Trovo poco utile nel ripassare le lezioni servirmi dei disegni, dei grafici o delle tabelle riassuntive contenute nel testo (71)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Esercitati ad utilizzare organizzatori semantici grafici (come schemi, tabelle, diagrammi, mappe concettuali). Ti possono essere utili per organizzare in maniera coerente e sistematica quanto studi, per facilitare la memorizzazione e favorire la capacità di risolvere problemi""",

    "C7": """C7 Autointerrogazione
Domanda: Poni domande all'insegnante, ai compagni e a te stesso per controllare la comprensione, il ricordo o la preparazione alle interrogazioni?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Quando mi preparo per un esame o una interrogazione, penso alle domande che l'insegnante potrà farmi (6)
Quando ho finito di studiare, immagino le domande che potrà farmi l'insegnante e cerco di rispondervi (25)
Prima di studiare un argomento cerco di chiarire che cosa si aspetta da me l'insegnante (35)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
In classe
Poni domande all'insegnante o ai tuoi compagni
Quando i tuoi compagni sono interrogati, ascolta le domande dell'insegnante e segnale sul quaderno
Quando studi
Pensa alle domande che l'insegnante potrebbe farti
Se nel testo sono previste domande, usale per comprendere e ricordare i concetti""",

    "C2": """C2 Autoregolazione
Domanda: Sei capace di gestire autonomamente lo studio e, in genere, i processi di apprendimento?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
All'inizio dello studio verifico quali sono le cose che devo fare (81)
Organizzo il mio studio in base al tempo che ho a disposizione (34)
Porto a termine in tempo utile i compiti da fare a casa (65)
Per stare più attento, durante le lezioni prendo degli appunti (12)
Dopo una lezione rivedo con cura i miei appunti per approfondire e ricordare meglio le idee raccolte (1)
Controllo se ho capito bene quello che l'insegnante ha detto durante la lezione (11)
Mentre studio mi pongo delle domande o faccio degli esercizi per verificare se ho capito bene (21)
Quando leggo rifletto sull'argomento e cerco di capire bene quello che è esposto nel testo (2)
Quando leggo un testo segno sul testo le cose più importanti (27)
Se ho un insuccesso, mi sento portato a ritentare l'impresa (80)
Quando eseguo un lavoro piuttosto noioso, penso ai suoi aspetti meno negativi e alla soddisfazione che proverò quando avrò finito (63)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Pianifica e organizza lo studio e i tuoi impegni in base al tempo che hai a disposizione
Tieni sotto controllo le tue azioni
Prendi appunti durante la spiegazione e sistemali dopo la lezione
Sottolinea e segna sul testo le cose più importanti""",

    "C3": """C3 Disorientamento
Domanda: Fai fatica ad orientarti nei vari compiti di studio e a organizzare le conoscenze?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Capita che riesco male in un compito perché non riesco a capire che cosa esattamente devo fare (3)
Mentre mi interrogano capita di accorgermi che ho studiato l'argomento sbagliato (40)
Trovo difficile capire se un concetto o un argomento mi risulta poco chiaro (98)
Mi capita di trovare che un argomento di studio era più difficile di quanto mi fossi aspettato (52)
Non riesco a rimanere concentrato nel lavoro più di un quarto d'ora (46)
Quando studio mi perdo nei dettagli e non riesco a trovare le cose principali (8)
Ho difficoltà a riassumere quanto ho ascoltato a scuola o letto in un libro (32)
Imparo a memoria regole, termini tecnici o formule, anche senza comprenderli (43)
Evito di fare domande, perché penso di dare fastidio all'insegnante (96)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Organizza e gestisci efficacemente:
il materiale da studiare
il tempo a disposizione
l'ambiente di studio""",

    "C6": """C6 Difficoltà di concentrazione
Domanda: Ti è difficile concentrarti nello studio e fatichi a organizzare tempi e spazi di lavoro?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Quando il mio insegnante spiega, mi trovo a pensare ad altre cose e così non seguo quello che sta esponendo (84)
Mentre studio mi distraggo facendo "sogni a occhi aperti”, progetti e programmi di ogni genere (89)
I problemi di casa o quelli posti dalle amicizie mi fanno trascurare l'impegno scolastico (79)
A casa studio le materie non in base a un piano preciso, ma secondo l'urgenza delle interrogazioni (69)
Quando mi accingo a studiare cerco di prevedere quanto tempo mi occorre per imparare un argomento (60)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Sviluppa valide ed efficaci strategie di controllo dell'attenzione
Evita un'eccessiva esposizione alla televisione, al computer e ai videogiochi
Elimina le fonti di distrazione
Poni e poniti spesso domande
Chiarisciti gli obiettivi che vuoi raggiungere con ciascuna attività e pianifica il tuo tempo""",

    "C4": """C4 Disponibilità alla collaborazione
Domanda: Apprezzi le occasioni in cui puoi studiare con altri? Sei disponibile ad uno studio partecipativo e collaborativo?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Trovo utile e stimolante discutere o lavorare in gruppo (30)
Quando partecipo a lavori di gruppo ho l'impressione di capire meglio le cose (50)
Mi sembra di imparare meglio quando posso confrontarmi con i miei compagni (74)
Penso che nello studio, come nel lavoro, è importante imparare a lavorare insieme (86)
Trovo che studiare con un compagno costituisca una perdita di tempo (13)
Quando partecipo a lavori di gruppo ho l'impressione di perdere il tempo (99)
Preferisco studiare la lezione da solo piuttosto che con l'aiuto altrui (57)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Prendi in considerazione l'importanza e l'utilità di studiare con altri al fine di:
comprendere meglio quanto studi
migliorare il tuo apprendimento
imparare a lavorare in gruppo""",

    "A1": """A1 Ansietà di base
Domanda: Fatichi a controllare l'ansietà e le tue reazioni emotive?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Quando devo affrontare un'interrogazione orale o un lavoro scritto sono così nervoso che non riesco ad esprimermi al meglio delle mie possibilità (28)
Quando sono interrogato all'improvviso, mi blocco e non riesco più a parlare (97)
Divento subito nervoso di fronte a una domanda o a un problema che non comprendo immediatamente (45)
Se mi accorgo di non avere più tempo per finire il lavoro sono preso dal panico (77)
Sono preso dal panico quando so che devo affrontare un esame scritto importante (23)
Il cuore mi batte forte quando devo subire un esame o un'interrogazione importante (38)
Mentre sto affrontando un'interrogazione la paura di sbagliare mi disturba così vado peggio (19)
Mi sento molto a disagio durante un lavoro scritto o un'interrogazione anche quando sono ben preparato (9)
Durante lo svolgimento di un compito in classe o durante un'interrogazione mi passano per la testa dubbi e incertezze sulla mia capacità di riuscire bene (33)
Quando prendo un brutto voto sono preso dallo scoraggiamento (4)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Rifletti sugli elementi che ti provocano un'ansia eccessiva per ricondurli ad una dimensione accettabile e gestibile
Ricorda che un certo livello di tensione interna è necessario per affrontare con la dovuta energia un compito impegnativo, ma un'eccessiva eccitazione nervosa può bloccare la tua risposta e ridurre le tue prestazioni
Hai bisogno di essere tranquillizzato, rassicurato e incoraggiato""",

    "A7": """A7 Interferenze emotive
Domanda: Non riesci a gestire le occasionali reazioni emotive che possono interferire nel lavoro scolastico?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Se per qualche motivo non riesco a preparare le lezioni, mi sento inquieto (55)
Se non riesco a prepararmi bene per la scuola, mi sento a disagio (66)
Se sono di cattivo umore mi concentro nello studio con difficoltà (87)
Se ho qualche problema emotivo (causato da cattivi rapporti con gli altri o con i genitori), non riesco ad applicarmi nello studio (92)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Rifletti sulle situazioni e sugli aspetti che possono provocarti reazioni emotive intense e un'inquietudine diffusa, per imparare a conoscere e gestire le tue emozioni e a vivere con serenità i tuoi impegni scolastici""",

    "A2": """A2 Volizione
Domanda: Sai gestire le attività scolastiche che richiedono impegno, sforzo e concentrazione? Riesci a portare a termine gli impegni e a raggiungere gli obiettivi che ti sei prefissato?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Anche se un compito è noioso, continuo a svolgerlo finché non l'ho terminato (67)
Mi impegno seriamente per conseguire un buon voto anche quando la materia non mi piace (54)
Quando incontro una difficoltà cerco di superarla, aumentando il mio impegno e la mia concentrazione (95)
Di fronte a un compito impegnativo, mi sento stimolato a sforzarmi di più (58)
Quando ho deciso di fare qualcosa, la porto a termine anche se costa fatica (70)
Vado a scuola avendo fatto tutti i compiti e studiato tutte le lezioni (49)
Quando per qualche ragione rimango indietro nel lavoro scolastico, cerco di colmare la lacuna senza che l'insegnante mi costringa a farlo (62)
Mi capita sia in casa che fuori casa di parlare con piacere delle cose che faccio a scuola (91)
Provo piacere quando devo svolgere un lavoro che mi impegna (42)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Prendere coscienza delle difficoltà che incontri nell'impegnarti nelle attività che richiedono sforzo, impegno e costanza e cercare di riconoscerne le cause per imparare a gestirle e a controllarle
Proteggere la tua motivazione da sollecitazioni e interessi alternativi e da stanchezza e frustrazione di fronte alle difficoltà che puoi incontrare
Rivedere il tuo atteggiamento verso la scuola e lo studio e riflettere sul valore che assegni agli obiettivi che ti poni""",

    "A5": """A5 Mancanza di perseveranza
Domanda: Fatichi a perseverare nello studio e a portare a termine i compiti assegnati?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Non appena incontro le prime difficoltà, abbandono un lavoro anche appena iniziato (75)
Se trovo che un argomento richiede tempo e fatica, non lo prendo neppure in considerazione (61)
Pensando alle cose che devo imparare mi capita di considerarle troppo difficili (82)
Quando non mi sento capace di completarlo, mi capita di lasciare a metà un lavoro già iniziato (53)
Quando mi va male qualcosa penso che ciò dipende dalle circostanze esterne più che dalla mia capacità o dal mio scarso impegno (76)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Rifletti e analizza le cause della mancanza di perseveranza che denota uno stato di demotivazione
Individua adeguate strategie di studio sia di natura cognitiva che motivazionale (es. proponiti obiettivi accessibili che puoi raggiungere in breve tempo)""",

    "A3": """A3 Attribuzione a cause controllabili
Domanda: Attribuisci le cause dei tuoi successi o fallimenti scolastici a fattori controllabili, come ad esempio il tuo sforzo o impegno?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Penso che la capacità di una persona dipende dalla costanza e dallo sforzo che mette nello studio (83)
Penso che la capacità di riuscire a scuola dipende dall'impegno che ciascuno mette nello studiare con cura (68)
Quando riesco a scuola, penso che ciò dipenda dall'aver studiato molto (5)
Quando mi va bene un'interrogazione penso che ho fatto proprio bene a studiare con tanto impegno (29)
Quando non riesco in un compito o in un'interrogazione penso che la ragione stia nel fatto che non ho studiato seriamente (15)
Penso che l'intelligenza di una persona può migliorare nel tempo, se questi si impegna seriamente (73)
Mi capita di pensare che se ci si impegna bene si può far crescere anche la propria intelligenza (94)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Diventa consapevole delle spiegazioni che attribuisci a successi e insuccessi tuoi e altrui
Prendi coscienza del fatto che la capacità di apprendere è data da un insieme di fattori che sono in evoluzione, non sono definiti una volta per tutte. Con l'esercizio e l'impegno puoi continuamente modificare tali fattori
Rivedi la tua idea di intelligenza: passa da una visione statica e non modificabile delle tue capacità scolastiche a una concezione dinamica dell'intelligenza, che può migliorare nel tempo se ti impegni con costanza""",

    "A4": """A4 Attribuzione a cause incontrollabili
Domanda: Attribuisci le cause dei tuoi successi o fallimenti scolastici a fattori incontrollabili, generalmente stabili e non modificabili, come ad esempio la fortuna o le domande facili o difficili del professore?
Se hai ottenuto un punteggio alto nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Quando riesco bene, penso che dipende dal fatto che il lavoro da svolgere era facile (51)
Quando vado bene in un'interrogazione, penso che l'insegnante è stato comprensivo e mi ha fatto domande facili (88)
Quando non riesco in un compito o in un'interrogazione penso che mi è stato chiesto qualcosa di troppo difficile (24)
Quando mi va bene un'interrogazione penso che per fortuna l'insegnante mi ha chiesto una cosa che sapevo (10)
Mi capita di pensare che gli insuccessi scolastici dipendono fondamentalmente dall'incapacità delle persone (47)
Mi capita di pensare che la capacità di riuscire a scuola dipenda dalle doti di intelligenza che uno ha (64)
Anche se mi impegno molto, mi viene da pensare che comunque non posso diventare più intelligente (59)
Mi capita di pensare che l'intelligenza di una persona è qualcosa che non può veramente cambiare: è un dono di natura (78)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Diventa consapevole delle spiegazioni che assegni a successi e insuccessi tuoi e altrui
Prendi coscienza che la capacità di apprendere è data da un insieme di fattori che sono in evoluzione, non sono definiti una volta per tutte. Con l'esercizio e l'impegno puoi continuamente modificare tali fattori
Rivedi la tua idea di intelligenza: passa da una visione statica e non modificabile delle tue capacità scolastiche a una concezione dinamica dell'intelligenza, che può migliorare nel tempo se ti impegni con costanza""",

    "A6": """A6 Percezione di competenza
Domanda: Ti consideri capace di riuscire a scuola? Pensi di essere responsabile nel portare a termine gli impegni di studio?
Se hai ottenuto un punteggio basso nel QSA
Item del QSA: Leggi attentamente gli item che si riferiscono a questo fattore:
Quando riesco a scuola, penso che dipende dal fatto che sono una persona veramente capace (39)
Quando mi va bene un'interrogazione, penso di essere proprio intelligente (20)
Quando inizio a svolgere un compito in classe, sono convinto di poter fare bene (14)
Mi sento sicuro di riuscire ad ottenere buoni voti (72)
Mi capita di pensare di essere capace di portare a termine con successo i miei impegni di studio (16)
Se sono preparato sono sicuro di riuscire bene in un compito o in un'interrogazione (93)
Suggerimenti: Quali strategie potresti mettere in atto per migliorare? Ecco alcuni esempi.
Ricorda che la percezione che hai di te è in stretta relazione con la motivazione e il successo scolastico
Sostieni sempre il circolo positivo tra i seguenti elementi: responsabilità verso il compito – soddisfazione per l'esecuzione del compito – stima di sé – percezione di competenza – potenziamento della consapevolezza delle proprie capacità nell’apprendimento – rinnovata responsabilità verso un nuovo compito più elevato – e così via"""
}


# Funzioni Python per i tool (invariate)
def tool_risultati(punteggi):
    session['punteggi'] = punteggi
    return f"Punteggi registrati: {json.dumps(punteggi)}"

def tool_af_qsa(fattori=None):
    if not fattori:
        fattori = list(AF_QSA_DATA.keys())
    punteggi = session.get('punteggi', {})
    output = ""
    for f in fattori:
        desc = AF_QSA_DATA.get(f, "Fattore non trovato.")
        punteggio = punteggi.get(f, "non specificato")
        output += f"{f}: Punteggio {punteggio}. {desc}\n\n"
    return output

def tool_sl_qsa():
    punteggi = session.get('punteggi', {})
    output = ""
    for gruppo, desc in SL_QSA_DATA.items():
        output += f"{gruppo}: {desc} (basato su punteggi: {json.dumps({k: punteggi.get(k) for k in desc.split(',')})})\n\n"
    return output

def tool_d_qsa(fattori=None):
    if not fattori:
        fattori = list(D_QSA_DATA.keys())
    output = ""
    for f in fattori:
        desc = D_QSA_DATA.get(f, "Scheda non trovata.")
        output += f"{desc}\n\n"
    return output

tool_functions = {
    "risultati": tool_risultati,
    "af-qsa": tool_af_qsa,
    "sl-qsa": tool_sl_qsa,
    "d-qsa": tool_d_qsa,
}

# Definizione dei tools per Ollama
tools = [
    {
        "type": "function",
        "function": {
            "name": "risultati",
            "description": "Registra i punteggi del QSA dell'utente",
            "parameters": {
                "type": "object",
                "properties": {
                    "punteggi": {
                        "type": "object",
                        "description": "Dizionario con i punteggi dei fattori (es. {'C1': 3.2, 'A1': 2.8})"
                    }
                },
                "required": ["punteggi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "af-qsa",
            "description": "Analisi fattore per fattore del QSA",
            "parameters": {
                "type": "object",
                "properties": {
                    "fattori": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista dei fattori da analizzare (es. ['C1', 'A1'])"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sl-qsa",
            "description": "Analisi di secondo livello del QSA con raggruppamenti tematici",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "d-qsa",
            "description": "Analisi approfondita dettagliata dei fattori QSA",
            "parameters": {
                "type": "object",
                "properties": {
                    "fattori": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista dei fattori da analizzare in dettaglio"
                    }
                }
            }
        }
    }
]

# Funzione per chiamare OpenRouter API
def call_openrouter(model, messages, tools=None):
    """Chiamata API OpenRouter per modelli gratuiti"""
    if not OPENROUTER_API_KEY:
        raise Exception("Chiave API OpenRouter non configurata")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Errore chiamata OpenRouter: {str(e)}")
        raise

def is_openrouter_model(model):
    """Controlla se il modello è di OpenRouter (contiene :free)"""
    return ":free" in model

# Funzione per generare risposta (usa client remoto e modello dinamico)
def generate_response(user_message, model):
    logger.info(f"🚀 Nuova richiesta ricevuta")
    logger.info(f"📝 Messaggio utente: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    logger.info(f"🤖 Modello selezionato: {model}")
    
    # Se il modello non è nella lista compatibile e l'utente sta fornendo punteggi, usa un modello compatibile
    if model not in TOOL_COMPATIBLE_MODELS and any(keyword in user_message.lower() for keyword in ['c1', 'c2', 'c3', 'a1', 'a2', 'a3', 'punteg', 'risultat', 'qsa']):
        original_model = model
        model = TOOL_COMPATIBLE_MODELS[0]  # Usa mixtral come default
        logger.info(f"🔄 Passaggio a modello compatibile con tools: {original_model} -> {model}")
    
    if 'messages' not in session:
        session['messages'] = []
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session['messages'] + [{"role": "user", "content": user_message}]
    logger.info(f"💬 Numero messaggi nella conversazione: {len(messages)}")
    
    try:
        # Determina se usare OpenRouter o Ollama
        use_openrouter = is_openrouter_model(model)
        logger.info(f"🔍 Provider: {'OpenRouter' if use_openrouter else 'Ollama'}")
        
        # Prima chiamata
        logger.info(f"🔄 Chiamando {'OpenRouter' if use_openrouter else 'Ollama'} per la prima volta...")
        
        if use_openrouter:
            response_data = call_openrouter(model, messages, tools)
            response = {
                'message': {
                    'content': response_data['choices'][0]['message']['content'],
                    'tool_calls': response_data['choices'][0]['message'].get('tool_calls', [])
                }
            }
        else:
            response = client.chat(model=model, messages=messages, tools=tools)
        
        logger.info(f"✅ Prima risposta ricevuta")
        
        # Log dettagli della risposta
        if 'message' in response:
            logger.info(f"📄 Contenuto risposta: {response['message'].get('content', 'N/A')[:150]}{'...' if len(response['message'].get('content', '')) > 150 else ''}")
            
            # Gestisci tool calls
            if 'tool_calls' in response.get('message', {}):
                logger.info(f"🛠️  Tool calls rilevati: {len(response['message']['tool_calls'])}")
                
                for i, tool_call in enumerate(response['message']['tool_calls']):
                    func_name = tool_call['function']['name']
                    args = tool_call['function'].get('arguments', {})
                    logger.info(f"🔧 Tool {i+1}: {func_name} con args: {str(args)[:100]}{'...' if len(str(args)) > 100 else ''}")
                    
                    if func_name in tool_functions:
                        try:
                            logger.info(f"⚙️  Eseguendo tool: {func_name}")
                            tool_output = tool_functions[func_name](**args)
                            logger.info(f"✅ Tool {func_name} eseguito con successo")
                            logger.info(f"📊 Output tool: {str(tool_output)[:200]}{'...' if len(str(tool_output)) > 200 else ''}")
                        except Exception as e:
                            logger.error(f"❌ Errore nel tool {func_name}: {str(e)}")
                            tool_output = f"Errore nel tool {func_name}: {str(e)}"
                    else:
                        logger.warning(f"⚠️  Tool non trovato: {func_name}")
                        tool_output = "Tool non trovato."
                    
                    messages.append({"role": "tool", "content": tool_output})
                
                # Seconda chiamata per risposta finale
                logger.info(f"🔄 Chiamando {'OpenRouter' if use_openrouter else 'Ollama'} per la seconda volta (dopo tools)...")
                
                if use_openrouter:
                    response_data = call_openrouter(model, messages)
                    response = {
                        'message': {
                            'content': response_data['choices'][0]['message']['content']
                        }
                    }
                else:
                    response = client.chat(model=model, messages=messages)
                    
                logger.info(f"✅ Seconda risposta ricevuta")
        
        final_content = response['message']['content']
        
        # Gestione fallback per modelli che restituiscono tool calls come testo
        if '[TOOL_CALLS]' in final_content and 'tool_calls' not in response.get('message', {}):
            logger.info(f"🔧 Rilevati tool calls nel testo, processamento manuale...")
            try:
                # Estrai il JSON dei tool calls dal testo con pattern più flessibili
                import re
                import json
                
                # Prova diversi pattern per estrarre i tool calls
                patterns = [
                    r'\[TOOL_CALLS\]\[(.*?)\]',
                    r'\[TOOL_CALLS\](.*?)(?:\n|$)',
                    r'TOOL_CALLS.*?(\{.*?\})',
                ]
                
                tool_data = None
                for pattern in patterns:
                    tool_match = re.search(pattern, final_content, re.DOTALL)
                    if tool_match:
                        try:
                            # Prova prima con json.loads
                            tool_data = json.loads(tool_match.group(1))
                            break
                        except:
                            try:
                                # Fallback con ast.literal_eval
                                import ast
                                tool_data = ast.literal_eval(tool_match.group(1))
                                break
                            except:
                                continue
                
                if tool_data:
                    func_name = tool_data.get('name')
                    args = tool_data.get('arguments', {})
                    
                    logger.info(f"🔧 Tool estratto: {func_name} con args: {str(args)[:100]}{'...' if len(str(args)) > 100 else ''}")
                    
                    if func_name in tool_functions:
                        try:
                            logger.info(f"⚙️  Eseguendo tool manualmente: {func_name}")
                            tool_output = tool_functions[func_name](**args)
                            logger.info(f"✅ Tool {func_name} eseguito con successo")
                            
                            # Aggiungi il risultato del tool ai messaggi
                            messages.append({"role": "tool", "content": tool_output})
                            
                            # Chiamata finale per ottenere la risposta elaborata
                            logger.info(f"🔄 Chiamando {'OpenRouter' if use_openrouter else 'Ollama'} per elaborare il risultato del tool...")
                            
                            if use_openrouter:
                                response_data = call_openrouter(model, messages)
                                final_content = response_data['choices'][0]['message']['content']
                            else:
                                response = client.chat(model=model, messages=messages)
                                final_content = response['message']['content']
                                
                            logger.info(f"✅ Risposta finale elaborata")
                            
                        except Exception as e:
                            logger.error(f"❌ Errore nell'esecuzione manuale del tool {func_name}: {str(e)}")
                            final_content = f"Ho ricevuto i tuoi punteggi ma c'è stato un errore nell'elaborazione: {str(e)}"
                    else:
                        logger.warning(f"⚠️  Tool non trovato nel processamento manuale: {func_name}")
                        final_content = "Ho ricevuto i tuoi punteggi ma il tool di analisi non è disponibile."
                else:
                    logger.warning(f"⚠️  Impossibile estrarre tool calls dal testo")
                    final_content = "Ho ricevuto i tuoi punteggi ma c'è stato un problema nell'estrazione dei dati. Potresti riprovare?"
                        
            except Exception as e:
                logger.error(f"❌ Errore nel processamento manuale dei tool calls: {str(e)}")
                final_content = "Ho ricevuto i tuoi punteggi ma c'è stato un errore nell'analisi. Potresti riprovare?"
        
        logger.info(f"🎯 Risposta finale generata: {len(final_content)} caratteri")
        
        # Aggiorna storia
        session['messages'].append({"role": "user", "content": user_message})
        session['messages'].append({"role": "assistant", "content": final_content})
        
        logger.info(f"💾 Conversazione aggiornata. Totale messaggi: {len(session['messages'])}")
        return final_content
        
    except Exception as e:
        logger.error(f"💥 ERRORE CRITICO durante la chiamata a Ollama: {str(e)}")
        logger.error(f"🔍 Tipo errore: {type(e).__name__}")
        import traceback
        logger.error(f"📍 Stack trace: {traceback.format_exc()}")
        return f"Errore durante la comunicazione con Ollama: {str(e)}"

# Route per la home (renderizza chat e passa la lista di modelli al template)
@app.route('/')
def index():
    logger.info(f"🏠 Accesso alla homepage da {request.remote_addr}")
    if 'messages' not in session:
        session['messages'] = []
    logger.info(f"💬 Messaggi in sessione: {len(session['messages'])}")
    return render_template('index.html', messages=session['messages'], models=MODELS)

# Route per inviare messaggio (AJAX)
@app.route('/send', methods=['POST'])
def send():
    logger.info(f"📨 Richiesta POST ricevuta da {request.remote_addr}")
    try:
        data = request.json
        user_message = data.get('message')
        model = data.get('model', MODELS[0])  # Default al primo modello se non specificato
        
        logger.info(f"📋 Dati ricevuti - Messaggio: {len(user_message) if user_message else 0} caratteri, Modello: {model}")
        
        if user_message:
            response = generate_response(user_message, model)
            logger.info(f"🎉 Risposta generata con successo")
            return jsonify({'response': response})
        else:
            logger.warning(f"⚠️  Messaggio vuoto ricevuto")
            return jsonify({'error': 'Messaggio vuoto'})
            
    except Exception as e:
        logger.error(f"💥 ERRORE nella route /send: {str(e)}")
        import traceback
        logger.error(f"📍 Stack trace: {traceback.format_exc()}")
        return jsonify({'error': f'Errore del server: {str(e)}'})

# Route per cancellare la cronologia (AJAX)
@app.route('/clear', methods=['POST'])
def clear_history():
    logger.info(f"🗑️  Richiesta di cancellazione cronologia da {request.remote_addr}")
    try:
        # Cancella tutti i messaggi dalla sessione
        session['messages'] = []
        # Cancella anche i punteggi registrati
        if 'punteggi' in session:
            del session['punteggi']
        
        logger.info(f"✅ Cronologia cancellata con successo")
        return jsonify({'success': True, 'message': 'Cronologia cancellata con successo'})
        
    except Exception as e:
        logger.error(f"💥 ERRORE nella cancellazione cronologia: {str(e)}")
        return jsonify({'success': False, 'error': f'Errore nella cancellazione: {str(e)}'})

# Route per iniziare una nuova chat (AJAX)
@app.route('/new_chat', methods=['POST'])
def new_chat():
    logger.info(f"🆕 Richiesta di nuova chat da {request.remote_addr}")
    try:
        # Cancella tutti i messaggi dalla sessione (stesso comportamento di clear)
        session['messages'] = []
        # Cancella anche i punteggi registrati
        if 'punteggi' in session:
            del session['punteggi']
        
        logger.info(f"✅ Nuova chat iniziata con successo")
        return jsonify({'success': True, 'message': 'Nuova chat iniziata'})
        
    except Exception as e:
        logger.error(f"💥 ERRORE nell'inizializzazione nuova chat: {str(e)}")
        return jsonify({'success': False, 'error': f'Errore nell\'inizializzazione: {str(e)}'})

if __name__ == '__main__':
    logger.info(f"🚀 Avvio server Flask sulla porta 5009")
    logger.info(f"🌐 URL: http://localhost:5009")
    logger.info(f"🤖 Modelli disponibili: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        logger.info(f"   {i}. {model}")
    
    app.run(debug=True, port=5009)
