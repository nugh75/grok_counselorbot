# Dati di Test per CounselorBot QSA

Ecco una serie di profili di test con punteggi QSA per testare tutte le funzionalità del chatbot:

## Profilo 1: Studente con Difficoltà Generalizzate
**Scenario**: Studente con basse competenze cognitive e problemi motivazionali
```json
{
  "C1": 2.1,
  "C2": 1.8,
  "C3": 7.2,
  "C4": 2.3,
  "C5": 1.9,
  "C6": 8.1,
  "C7": 2.0,
  "A1": 7.8,
  "A2": 2.2,
  "A3": 2.1,
  "A4": 7.5,
  "A5": 7.9,
  "A6": 1.7,
  "A7": 8.0
}
```

## Profilo 2: Studente Eccellente
**Scenario**: Studente con ottime competenze e alta motivazione
```json
{
  "C1": 8.5,
  "C2": 8.2,
  "C3": 2.1,
  "C4": 7.8,
  "C5": 8.0,
  "C6": 1.9,
  "C7": 8.3,
  "A1": 2.0,
  "A2": 8.7,
  "A3": 8.5,
  "A4": 1.8,
  "A5": 2.2,
  "A6": 8.4,
  "A7": 1.7
}
```

## Profilo 3: Studente con Ansia da Prestazione
**Scenario**: Buone competenze cognitive ma alta ansia e interferenze emotive
```json
{
  "C1": 7.2,
  "C2": 6.8,
  "C3": 3.5,
  "C4": 5.9,
  "C5": 7.1,
  "C6": 4.2,
  "C7": 6.5,
  "A1": 8.3,
  "A2": 6.2,
  "A3": 6.8,
  "A4": 4.1,
  "A5": 4.7,
  "A6": 5.9,
  "A7": 8.5
}
```

## Profilo 4: Studente con Problemi di Concentrazione
**Scenario**: Discrete competenze ma gravi difficoltà di concentrazione e orientamento
```json
{
  "C1": 5.8,
  "C2": 4.2,
  "C3": 8.7,
  "C4": 6.1,
  "C5": 5.5,
  "C6": 8.9,
  "C7": 4.8,
  "A1": 5.2,
  "A2": 5.8,
  "A3": 5.4,
  "A4": 6.1,
  "A5": 6.8,
  "A6": 4.9,
  "A7": 6.2
}
```

## Profilo 5: Studente Individualista
**Scenario**: Buone competenze individuali ma scarsa collaborazione
```json
{
  "C1": 7.8,
  "C2": 7.5,
  "C3": 2.8,
  "C4": 1.9,
  "C5": 6.9,
  "C6": 2.5,
  "C7": 6.2,
  "A1": 3.1,
  "A2": 7.2,
  "A3": 7.8,
  "A4": 2.7,
  "A5": 3.2,
  "A6": 7.6,
  "A7": 2.9
}
```

## Profilo 6: Studente Medio con Aree Specifiche
**Scenario**: Competenze medie con alcuni punti di forza e debolezza specifici
```json
{
  "C1": 4.8,
  "C2": 5.2,
  "C3": 5.1,
  "C4": 6.8,
  "C5": 3.9,
  "C6": 5.8,
  "C7": 4.5,
  "A1": 4.7,
  "A2": 5.9,
  "A3": 4.2,
  "A4": 5.3,
  "A5": 5.7,
  "A6": 5.1,
  "A7": 4.8
}
```

## Profilo 7: Studente con Bassa Autostima
**Scenario**: Discrete competenze ma bassa percezione di competenza e attribuzione esterna
```json
{
  "C1": 6.2,
  "C2": 5.8,
  "C3": 4.5,
  "C4": 5.4,
  "C5": 6.1,
  "C6": 3.8,
  "C7": 5.7,
  "A1": 6.8,
  "A2": 4.9,
  "A3": 2.8,
  "A4": 7.9,
  "A5": 6.5,
  "A6": 2.1,
  "A7": 5.2
}
```

---

## ⚙️ Modelli Compatibili con Tool Calls

**IMPORTANTE**: Per testare correttamente il chatbot QSA, devi usare SOLO questi modelli che supportano i tool calls:

### 🔧 Modelli Ollama Compatibili:
- `mixtral:8x7b` ⭐ **RACCOMANDATO**
- `qwen2.5:7b`
- `qwen3:32b`
- `phi4:14b`
- `llama3:latest`

### 🌐 Modelli OpenRouter Compatibili:
- `microsoft/phi-3-medium-128k-instruct:free` ⭐ **RACCOMANDATO**
- `mistralai/mistral-7b-instruct:free`
- `deepseek/deepseek-chat-v3-0324:free`
- `meta-llama/llama-3-8b-instruct:free`

### ❌ Modelli NON Compatibili (NON usare per i test):
- `deepseek-r1:latest` (Ollama - non supporta tool calls)
- `llama4:16x17b` (Ollama - non supporta tool calls)
- `devstral:24b` (Ollama - non supporta tool calls)
- `qwen:110b` (Ollama - non supporta tool calls)
- `microsoft/phi-3-mini-128k-instruct:free` (OpenRouter - limitato)
- Tutti gli altri modelli OpenRouter free non elencati sopra

## Come Usare i Dati di Test

### Test Rapido (formato messaggi)
Puoi copiare e incollare questi messaggi nella chat:

**Per Profilo 1:**
```
I miei punteggi QSA sono: C1=2.1, C2=1.8, C3=7.2, C4=2.3, C5=1.9, C6=8.1, C7=2.0, A1=7.8, A2=2.2, A3=2.1, A4=7.5, A5=7.9, A6=1.7, A7=8.0
```

**Per Profilo 2:**
```
I miei punteggi QSA sono: C1=8.5, C2=8.2, C3=2.1, C4=7.8, C5=8.0, C6=1.9, C7=8.3, A1=2.0, A2=8.7, A3=8.5, A4=1.8, A5=2.2, A6=8.4, A7=1.7
```

**Per Profilo 3:**
```
I miei punteggi QSA sono: C1=7.2, C2=6.8, C3=3.5, C4=5.9, C5=7.1, C6=4.2, C7=6.5, A1=8.3, A2=6.2, A3=6.8, A4=4.1, A5=4.7, A6=5.9, A7=8.5
```

### Test dei Colori e Fattori Invertiti

I profili sono stati creati per testare:

1. **Fattori Normali** (più alto = meglio): C1, C2, C4, C5, C7, A2, A3, A6
2. **Fattori Invertiti** (più alto = peggio): C3, C6, A1, A4, A5, A7

### Categorie di Colore da Testare:

- **VERDE (6-9)**: Punteggi alti per fattori normali, bassi per fattori invertiti
- **GIALLO (4-5)**: Punteggi medi
- **ARANCIONE (1-3)**: Punteggi bassi per fattori normali, alti per fattori invertiti

### Scenari di Test Consigliati:

1. **Test Funzionalità Base**: Usa Profilo 6 (valori medi)
2. **Test Analisi Critica**: Usa Profilo 1 (molti problemi)
3. **Test Positivo**: Usa Profilo 2 (studente eccellente)
4. **Test Casi Specifici**: Usa Profili 3, 4, 5, 7 per scenari particolari
5. **Test Tool Calls**: Prova tutti i profili per verificare che i tool vengano chiamati correttamente
6. **Test Interpretazione Colori**: Verifica che il sistema interpreti correttamente i fattori invertiti

### 🧪 Procedura di Test Corretta:

1. **Seleziona un modello compatibile** (preferibilmente `mixtral:8x7b` o `microsoft/phi-3-medium-128k-instruct:free`)
2. **Copia uno dei messaggi di test** (ad esempio per Profilo 1)
3. **Incolla nella chat** e invia
4. **Verifica che il chatbot**:
   - Chiami il tool `risultati` per registrare i punteggi
   - Chiami il tool `af-qsa` per l'analisi fattore per fattore
   - Fornisca un'analisi completa con interpretazione dei colori
   - NON restituisca solo il JSON dei punteggi

### ⚠️ Risoluzione Problemi:

**Se il chatbot restituisce solo il JSON** (come `{"C1":2.1,"C2":1.8,...}`):
- Stai usando un modello non compatibile
- Cambia modello e riprova
- Verifica che il modello sia nella lista compatibile sopra

**Se non vedi analisi dettagliata**:
- Il modello potrebbe non supportare correttamente i tool calls
- Prova con `mixtral:8x7b` (più affidabile)
