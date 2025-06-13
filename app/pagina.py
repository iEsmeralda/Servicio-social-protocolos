# Importando las Librerias
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re

# Modelos para los embeddings
MODELOS = {
    "sentence-transformers": 'Santp98/SBERT-pairs-bert-base-spanish-wwm-cased',
    "sentence_similarity": 'hiiamsid/sentence_similarity_spanish_es'
}
CAMPOS = ["Titulo", "resumen", "objetivos", "claves"]

# Extracción de NER's
def cargar_modelo_ner(nombre_modelo="iEsmeralda/mrm8488-finetuned-ner-tech"):
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForTokenClassification.from_pretrained(nombre_modelo)
    ner_pipeline = pipeline("ner", model=modelo, tokenizer=tokenizador, aggregation_strategy="simple")
    # aggregation_strategy="simple" indica que se deben agrupar los tokens que pertenezcan a una misma entidad, por ejemplo si el modelo detecta "inteligencia" y "artificial"
    # como parte de una misma entidad, los junta en una sola: "inteligencia artificial"

    return ner_pipeline

# funcion para agrupar entidades consecutivas del mismo tipo, esto se hizo porque habia entidades que no fueron "unidas" a pesar del aggregation_strategy
def agrupar_entidades_consecutivas(entidades):
    # si no hay entidades, se regresa una lista vacia
    if not entidades:
        return []
    entidades_agrupadas = []
    entidad_actual = entidades[0].copy() # se toma la primera entidad como "base"
    for i in range(1, len(entidades)):
        entidad = entidades[i]

        # si la entidad es del mismo tipo y esta justo despues de la actual, entonces:
        if entidad["entity_group"] == entidad_actual["entity_group"] and entidad["start"] == entidad_actual["end"] + 1:
            # se une la palabra, se actualiza el fin y se promedia el puntaje de ambas entidades
            entidad_actual["word"] += " " + entidad["word"]
            entidad_actual["end"] = entidad["end"]
            entidad_actual["score"] = (entidad_actual["score"] + entidad["score"]) / 2
        else:
            # si no son consecutivas, se guarda la actual y se pasa a la siguiente entidad
            entidades_agrupadas.append(entidad_actual)
            entidad_actual = entidad.copy()
    entidades_agrupadas.append(entidad_actual)
    return entidades_agrupadas

# funcion para extraer las entidades nombradas de un texto largo
def extraer_ners(texto, ner_pipeline, max_tokens=512):
    tokenizador = ner_pipeline.tokenizer
    modelo = ner_pipeline.model

    tokens = tokenizador.tokenize(texto)
    entidades = []

    # se recorre la lista completa de tokens en bloques de tamaño max_tokens ya que la mayoria de los modelos de transformers solo aceptan secuencias de hasta 512 tokens como maximo
    for i in range(0, len(tokens), max_tokens):
        bloque_de_tokens = tokens[i:i+max_tokens]

        # la lista de tokens nombrada como bloque_de_tokens se pasa a texto plano porque el pipeline de ner espera un texto como entrada
        bloque_texto = tokenizador.convert_tokens_to_string(bloque_de_tokens)

        # se vuelve a tokenizar el texto del bloque para asegurarse que no se pase del limite real de tokens del modelo
        if len(tokenizador(bloque_texto)["input_ids"]) > max_tokens:
            # si el bloque excede el limite de 512 tokens, se salta
            continue

        # se aplica el pipeline de ner al bloque de texto para obtener las entidades detectadas en ese bloque
        entidades_en_bloque = ner_pipeline(bloque_texto)
        entidades_agrupadas_bloque = agrupar_entidades_consecutivas(entidades_en_bloque)
        entidades.extend(entidades_agrupadas_bloque)

    return [entidad["word"] for entidad in entidades]

# Lectura del archivo de protocolos
protocolos=pd.read_csv("app/protocolos_completo_limpios.csv")
protocolos_acentuados=pd.read_csv("app/protocolos_completo_limpios.csv")

# Lectura del archivo de profesores
profesores=pd.read_csv("app/profesores_completos_2023.csv")

# Consulta de los protocolos similares
def buscarProtocolos(consulta):
    df_resultados_acentuados  = protocolos_acentuados
    ner_pipeline = cargar_modelo_ner()

    query = consulta.strip()

    #print(f"\nConsulta original: {query}")
    entidades_query = extraer_ners(query, ner_pipeline)
    #print("Entidades NER detectadas:", entidades_query)

    # si hay entidades, se enriquece un texto
    query_enriquecida = f"{query} {' '.join(entidades_query)}" if entidades_query else query
    query_con_contexto = f"Este trabajo trata sobre {query_enriquecida}" # se agrega contexto para el modelo, pues si solo ponemos la entidad, el modelo no sabe de que se trata

    df_resultados_acentuados = protocolos_acentuados.copy()
    df_resultados_acentuados["sim_total"] = 0

    for clave_modelo, nombre_modelo in MODELOS.items():
        modelo = SentenceTransformer(nombre_modelo)
        # se genera el embedding de la consulta con contexto
        embedding_query = modelo.encode(query_con_contexto, convert_to_tensor=False)

        for campo in CAMPOS:
            # se construye el nombre del archivo .pkl que contiene los embeddings del campo
            nombre_pkl = f"{campo}_{clave_modelo}_embeddings.pkl"
            ruta_pkl = os.path.join("app/pkl", nombre_pkl)
            if not os.path.exists(ruta_pkl):
                print(f"Falta el archivo: {ruta_pkl}")
                continue

            with open(ruta_pkl, "rb") as f:
                embeddings_cargados = pickle.load(f)

            # se calcula la similitud coseno entre el embedding de la consulta y los del archivo
            simi_coseno = cosine_similarity([embedding_query], embeddings_cargados)[0]
            df_resultados_acentuados[f"{campo}_{clave_modelo}"] = simi_coseno
            # se acumula la similitud en la columna "sim_total" (ponderacion simple)
            df_resultados_acentuados["sim_total"] += simi_coseno

    df_resultados_acentuados = df_resultados_acentuados.sort_values("sim_total", ascending=False).head(10)

    # Crear el diccionario con listas vacías para acumular los valores
    diccionario_resultados = {
        "#": [],
        "TT": [],
        "Título": [],
        "Similitud": [],
        "Resumen": [],
        "Directores": [],
        "Claves": []
    }

    print("\nTop 10 resultados más similares:")
    for idx, (i, row) in enumerate(df_resultados_acentuados.iterrows(), start=1):
        print(f"TT: {row['TT']}")
        print(f"Título: {row['Titulo']}")
        print(f"Similitud total: {row['sim_total']:.4f}")
        
        # Llenar el diccionario
        diccionario_resultados["#"].append(idx)
        diccionario_resultados["TT"].append(row["TT"])
        diccionario_resultados["Título"].append(row["Titulo"])
        diccionario_resultados["Similitud"].append(round(row["sim_total"], 4))
        diccionario_resultados["Resumen"].append(row["resumen"])
        diccionario_resultados["Directores"].append(row["directores"])
        diccionario_resultados["Claves"].append(row["claves"])

    return diccionario_resultados

def quitar_acentos(texto):
    if not isinstance(texto, str):
        return ''
    # Quitar acentos
    texto_sin_acentos = ''.join(
        c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'
    )
    # Eliminar todo excepto letras, espacios y comas
    texto_limpio = re.sub(r'[^a-zA-Z\s,]', '', texto_sin_acentos)
    return texto_limpio.lower().strip()

def obtener_info_directores(tt_actual, nombres_directores, df_profesores):
    info_directores = []
    
    nombres_directores = [n.strip() for n in nombres_directores.split(',')]
    nombres_directores_sin_acentos = [quitar_acentos(n).lower() for n in nombres_directores]
    
    for index, fila in df_profesores.iterrows():
        direcciones = str(fila['Direcciones']).split(',')
        
        if any(tt_actual in d for d in direcciones):
            nombre_profesor = str(fila['Nombre_normalizado'])
            nombre_profesor_sin_acentos = quitar_acentos(nombre_profesor).lower()
            
            for nombre_sin_acentos in nombres_directores_sin_acentos:
                if nombre_sin_acentos in nombre_profesor_sin_acentos:
                    info = f'''
                        <p><strong>Nombre:</strong> {nombre_profesor.title()}</p>
                        <p><strong>Unidades Académicas impartidas:</strong> {fila["UAs_impartidas"]}</p>
                        <p><strong>Departamento:</strong> {fila["DEPTO"]}</p>
                        <p><strong>Academia:</strong> {fila["ACADEMIA"]}</p>
                        <p><strong>Estudios e intereses:</strong> {fila["Estudios_intereses"]}</p>
                        <hr>
                    '''
                    info_directores.append(info)
    
    if not info_directores:
        return "<p>No se encontró información de los directores.</p>"
    
    return "".join(info_directores)

# -------------------------------------------- Página web --------------------------------------------------

# Se importa la librería (framework) que permite desarrollar aplicaciones web
from flask import Flask, render_template, request, jsonify

# Se crea una instancia de la clase Flask, con una carpeta estatica para css e img
app = Flask(__name__, static_folder='static')

# Se definen las rutas y vistas: Define una ruta URL y una función de vista asociada a esa ruta, la cual será ejecutada cuando se acceda a la ruta especificada:
@app.route('/') # La ruta "/" es la ruta raíz de la página web.
def index():
    # Se mostrará el archivo index.html cuando se accreda a esta ruta
    return render_template('index.html')

@app.route('/index')
def inicio():
    # Se mostrará el archivo index.html cuando se accreda a esta ruta
    return render_template('index.html')

# Para la conexión con el archivo de JavaScript
@app.route('/app/static/js/script.js')
def servir_script():
    return app.send_static_file('script.js')

# Recomendaciones
@app.route('/resultadosProtocolos', methods=['POST'])
def obtener_recomendaciones():
    query = request.form['consulta'] # Ejemplo: Sistema de Monitoreo de Patineta Eléctrica 

    # Llamar a la función buscarProtocolos que devuelve el diccionario con los resultados
    diccionario_resultados = buscarProtocolos(consulta=query)

    # Construcción del contenido HTML de la tabla
    contenido_html = '''
    <div class="table-responsive">
      <table class="table table-bordered tablaResultados">
        <thead class="table-primary">
          <tr>
            <th scope="col">#</th>
            <th scope="col">Título</th>
            <th scope="col">Similitud</th>
            <th scope="col">Información <i class="fa-regular fa-file-lines"></i></th>
          </tr>
        </thead>
        <tbody class="table-light">
    '''
    
    # Recorrer los datos y construir filas con modales individuales
    for i in range(10):
        modal_id = f"modal{i}"  # id único para cada modal
        ruta_pdf = f'app/static/pdf/{diccionario_resultados["TT"][i]}.pdf' #Ruta general de los pdf
        existe_pdf = os.path.exists(ruta_pdf)
        # Generar botón PDF solo si existe
        boton_pdf = f'<a href="static/pdf/{diccionario_resultados["TT"][i]}.pdf" target="_blank" class="btn btn-info">Ver PDF</a>' if existe_pdf else '<span class="text-muted">PDF no disponible</span>'

        # Llamar a la función para obtener la información de los directores
        info_directores_modal = obtener_info_directores(diccionario_resultados["TT"][i], diccionario_resultados["Directores"][i], profesores)

        contenido_html += f'''
        <tr>
            <th scope="row">{diccionario_resultados["#"][i]}</th>
            <td>{diccionario_resultados["Título"][i]}</td>
            <td>{diccionario_resultados["Similitud"][i]}</td>
            <td>
                <button type="button" class="btn btn-primary btn-sm" data-bs-toggle="modal" data-bs-target="#{modal_id}">
                    Ver detalles
                </button>
            </td>
        </tr>

        <!-- Modal para la fila {i} -->
        <div class="modal fade" id="{modal_id}" tabindex="-1" aria-labelledby="label{modal_id}" aria-hidden="true">
            <div class="modal-dialog modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                <h5 class="modal-title" id="label{modal_id}">{diccionario_resultados["Título"][i]}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                <p><strong>Resumen:</strong> {diccionario_resultados["Resumen"][i]}</p>
                <p><strong>Directores:</strong> {diccionario_resultados["Directores"][i]}</p>
                <p><strong>Palabras clave:</strong> {diccionario_resultados["Claves"][i]}</p>
                <p><strong>Año:</strong> {diccionario_resultados["TT"][i][:4]}</p>
                </div>
                <div class="modal-footer">
                {boton_pdf}
                <button class="btn btn-info" data-bs-target="#{modal_id}_2" data-bs-toggle="modal">Directores</button>
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                </div>
            </div>
            </div>
        </div>

        <div class="modal fade" id="{modal_id}_2" aria-hidden="true" aria-labelledby="exampleModalToggleLabel2" tabindex="-1">
            <div class="modal-dialog modal-dialog-scrollable">
                <div class="modal-content">
                <div class="modal-header">
                    <h1 class="modal-title fs-5" id="exampleModalToggleLabel2">Información de los directores</h1>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    {info_directores_modal}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" data-bs-target="#{modal_id}" data-bs-toggle="modal">Regresar</button>
                </div>
                </div>
            </div>
        </div>
        '''

    contenido_html += '''
        </tbody>
      </table>
    </div>
    <p><br>Esperamos que estos resultados sean de utilidad.</p>
    '''

    return contenido_html

# -----------------

# Código para ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)