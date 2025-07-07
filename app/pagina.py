# Importando las Librerias
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, session, jsonify, redirect, url_for, flash
import secrets
import re, smtplib
from email.message import EmailMessage
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import uuid
import sys
import unicodedata
import shutil
import threading
import webbrowser
import socket


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
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
pdfs_disponibles = []
pdf_folder = os.path.join(base_path, 'static', 'pdf')

try:
    pdfs_disponibles = os.listdir(pdf_folder)
except Exception as e:
    print("No se pudieron listar los PDFs:", e)
protocolos=pd.read_csv(os.path.join(base_path, 'protocolos_completo_limpios.csv'))
protocolos_acentuados=pd.read_csv(os.path.join(base_path, 'protocolos_completo_limpios.csv'))

# Lectura del archivo de profesores
profesores=pd.read_csv(os.path.join(base_path, 'profesores_completos_2023.csv'))

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

def contiene_subfrase_en_orden(nombre_completo, patron):
    # Verifica si la secuencia de palabras en 'patron' aparece en orden en 'nombre_completo'.
    palabras_completas = nombre_completo.split()
    palabras_patron = patron.split()
    
    for i in range(len(palabras_completas) - len(palabras_patron) + 1):
        if palabras_completas[i:i + len(palabras_patron)] == palabras_patron:
            return True
    return False

# Consulta de los protocolos similares
def buscarProtocolos(consulta, nombreProfesor=None, anioTT=None):
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
            ruta_pkl = os.path.join(base_path, 'pkl', nombre_pkl)
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

    df_similitud_global = df_resultados_acentuados.copy() # Copia por si al filtrar no queda nada

    # Se realizan las comparativas para filtrar el profesor y año de TT si existen
    if nombreProfesor:
        patron = quitar_acentos(nombreProfesor.lower())
        df_resultados_acentuados = df_resultados_acentuados[
            df_resultados_acentuados.apply(
                lambda row: contiene_subfrase_en_orden(quitar_acentos(str(row.get("director1", ""))), patron)
                            or contiene_subfrase_en_orden(quitar_acentos(str(row.get("director2", ""))), patron),
                axis=1
            )
        ]
    if anioTT:
        df_resultados_acentuados = df_resultados_acentuados[
            df_resultados_acentuados["TT"].astype(str).str[:4] == str(anioTT)
        ]

    # Si por alguna razón el df queda vacío despues de los filtros entonces se obtienen los resultados sin filtros
    if df_resultados_acentuados.empty:
        df_resultados_acentuados = df_similitud_global

    df_resultados_acentuados = df_resultados_acentuados.sort_values("sim_total", ascending=False).head(10)

    # Crear el diccionario con listas vacías para acumular los valores
    diccionario_resultados = {
        "#": [],
        "TT": [],
        "Título": [],
        "Similitud": [],
        "Resumen": [],
        "Directores": [],
        "Claves": [],
        "link_pdf": []
    }

    print("\nTop 10 resultados más similares:")
    for idx, (i, row) in enumerate(df_resultados_acentuados.iterrows(), start=1):
        print(f"TT: {row['TT']}")
        print(f"Título: {row['Titulo']}")
        print(f"Similitud total: {row['sim_total']:.4f}")
        print(nombreProfesor, anioTT)

        # Construir nombre de directores
        directores = []
        if pd.notna(row.get("director1")) and str(row["director1"]).strip():
            directores.append(str(row["director1"]).strip())
        if pd.notna(row.get("director2")) and str(row["director2"]).strip():
            directores.append(str(row["director2"]).strip())
        directores_concatenados = ", ".join(directores)
        
        # Llenar el diccionario
        diccionario_resultados["#"].append(idx)
        diccionario_resultados["TT"].append(row["TT"])
        diccionario_resultados["Título"].append(row["Titulo"])
        diccionario_resultados["Similitud"].append(round(row["sim_total"], 4))
        diccionario_resultados["Resumen"].append(row["resumen"])
        diccionario_resultados["Directores"].append(directores_concatenados)
        diccionario_resultados["Claves"].append(row["claves"])
        nombre_pdf = f"{row['TT']}.pdf"
        link_pdf = f"/static/pdf/{nombre_pdf}" if nombre_pdf in pdfs_disponibles else None
        diccionario_resultados["link_pdf"].append(link_pdf)

    return diccionario_resultados

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
                        <p><strong>Contacto:</strong> {fila["correo"]}</p>
                        <hr>
                    '''
                    info_directores.append(info)
    
    if not info_directores:
        return "<p>No se encontró información de los directores.</p>"
    
    return "".join(info_directores)

# -------------------------------------------- Página web --------------------------------------------------

# Se importa la librería (framework) que permite desarrollar aplicaciones web
key = secrets.token_urlsafe(64)

# Se crea una instancia de la clase Flask, con una carpeta estatica para css e img
app = Flask(__name__, static_folder='static')
app.secret_key = key

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
    query = request.form['consulta']
    diccionario_resultados = buscarProtocolos(consulta=query)

    contenido_html = '''
    <div class="table-responsive">
      <table class="table table-bordered tablaResultados">
        <thead class="table-primary">
          <tr>
            <th>#</th><th>Título</th><th>Similitud</th><th>Información</th>
          </tr>
        </thead>
        <tbody class="table-light">
    '''

    for idx in range(len(diccionario_resultados["TT"])):
        modal_id      = f"modal{idx}"
        tt            = diccionario_resultados["TT"][idx]
        titulo        = diccionario_resultados["Título"][idx]
        sim           = diccionario_resultados["Similitud"][idx]
        resumen       = diccionario_resultados["Resumen"][idx]
        directores    = diccionario_resultados["Directores"][idx]
        claves        = diccionario_resultados["Claves"][idx]
        nombre_pdf    = f"{tt}.pdf"
        existe_pdf    = nombre_pdf in pdfs_disponibles

        # 1) Botón “Ver detalles”
        if 'usuario' in session:
            boton_info = (f'<button class="btn btn-primary btn-sm" '
                          f'data-bs-toggle="modal" data-bs-target="#{modal_id}">'
                          'Ver detalles</button>')
        else:
            boton_info = (f'<a href="{url_for("login")}" class="btn btn-primary btn-sm">'
                          'Ver detalles</a>')

        # 2) Botón “Ver PDF”
        if 'usuario' in session:
            if existe_pdf:
                boton_pdf = (f'<a href="/static/pdf/{nombre_pdf}" '
                             'target="_blank" class="btn btn-info">Ver PDF</a>')
            else:
                boton_pdf = '<span class="text-muted">PDF no disponible</span>'
        else:
            boton_pdf = (f'<a href="{url_for("login")}" '
                         'class="btn btn-outline-secondary btn-sm">'
                         'Inicia sesión para ver PDF</a>')

        # 3) Fila de la tabla
        contenido_html += f'''
        <tr>
          <td>{idx+1}</td>
          <td>{titulo}</td>
          <td>{sim}</td>
          <td>{boton_info}</td>
        </tr>
        '''

        # 4) Modal de detalles (solo si está logueado)
        if 'usuario' in session:
            info_directores_modal = obtener_info_directores(tt, directores, profesores)
            contenido_html += f'''
            <div class="modal fade" id="{modal_id}" tabindex="-1" aria-labelledby="label{modal_id}" aria-hidden="true">
              <div class="modal-dialog modal-dialog-scrollable">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title" id="label{modal_id}">{titulo}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                  </div>
                  <div class="modal-body">
                    <p><strong>Resumen:</strong> {resumen}</p>
                    <p><strong>Directores:</strong> {directores}</p>
                    <p><strong>Claves:</strong> {claves}</p>
                    <p><strong>Año:</strong> {tt[:4]}</p>
                  </div>
                  <div class="modal-footer">
                    {boton_pdf}
                    <button class="btn btn-info" data-bs-target="#{modal_id}_2" data-bs-toggle="modal">Directores</button>
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                  </div>
                </div>
              </div>
            </div>
            <div class="modal fade" id="{modal_id}_2" tabindex="-1">
              <div class="modal-dialog modal-dialog-scrollable">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title">Información de los directores</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
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
    <p>Esperamos que estos resultados sean de utilidad.</p>
    '''
    return contenido_html

# -----------------
# Ruta interna del .exe (solo lectura)
origen_db = os.path.join(base_path, 'usuarios.db')

# Ruta externa (escribible): junto al ejecutable
destino_db = os.path.join(os.getcwd(), 'usuarios.db')

# creamos la base de datos de usuarios
def crear_db():

    
    if not os.path.exists(destino_db):
        shutil.copyfile(origen_db, destino_db)

    conexion = sqlite3.connect(destino_db)
    cursor = conexion.cursor()
    cursor.execute('''
    CREATE TABLE if not exists usuarios
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    correo TEXT UNIQUE NOT NULL,
    contrasena TEXT NOT NULL)
    ''')
    conexion.commit()
    conexion.close()

crear_db()

@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        correo = request.form['correo'].strip().lower()
        contrasena = request.form['contrasena']
        # Solo validamos dominio
        if not re.match(r".+@(alumno\.ipn\.mx|ipn\.mx|alumnoguinda\.mx)$", correo):
            flash("Solo se permiten correos institucionales (@ipn.mx, @alumno.ipn.mx, @alumnoguinda.mx)", "danger")
            return redirect(url_for('registro'))

        hash_pass = generate_password_hash(contrasena)
        if not os.path.exists(destino_db):
            shutil.copyfile(origen_db, destino_db)

        conexion = sqlite3.connect(destino_db)
        c = conexion.cursor()
        try:
            c.execute(
                "INSERT INTO usuarios (correo, contrasena) VALUES (?, ?)",
                (correo, hash_pass)
            )
            conexion.commit()
            flash("Usuario registrado correctamente. Ahora puedes iniciar sesión.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Ese correo ya está registrado.", "warning")
            return redirect(url_for('registro'))
        finally:
            conexion.close()

    return render_template('registro.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        correo = request.form['correo'].strip().lower()
        contrasena = request.form['contrasena']
        if not os.path.exists(destino_db):
            shutil.copyfile(origen_db, destino_db)

        conexion = sqlite3.connect(destino_db)
        c = conexion.cursor()
        c.execute(
            "SELECT contrasena FROM usuarios WHERE correo=?",
            (correo,)
        )
        row = c.fetchone()
        conexion.close()

        if row and check_password_hash(row[0], contrasena):
            session['usuario'] = correo
            flash("Sesión iniciada exitosamente.", "success")
            return redirect(url_for('index'))
        else:
            flash("Credenciales inválidas.", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('usuario', None)
    flash("Has cerrado sesión.", "info")
    return redirect(url_for('index'))

# Recomendaciones filtradas
@app.route('/resultadosProtocolosConFiltro', methods=['POST'])
def obtener_recomendaciones_filtradas():
    query           = request.form['consulta']
    profesor        = request.form['profesor']
    anio            = request.form['anio']
    anio_TT         = anio.strip() if anio else None
    nombre_Profesor = profesor.strip() if profesor else None

    # Buscamos resultados ya filtrados
    dicc = buscarProtocolos(
        consulta=query,
        nombreProfesor=nombre_Profesor,
        anioTT=anio_TT
    )

    html = ['''
    <div class="table-responsive">
      <table class="table table-bordered tablaResultados">
        <thead class="table-primary">
          <tr>
            <th>#</th><th>Título</th><th>Similitud</th><th>Información</th>
          </tr>
        </thead>
        <tbody class="table-light">
    ''']

    for idx in range(len(dicc["TT"])):
        modal_id   = f"modal{idx}"
        tt         = dicc["TT"][idx]
        titulo     = dicc["Título"][idx]
        sim        = dicc["Similitud"][idx]
        resumen    = dicc["Resumen"][idx]
        directores = dicc["Directores"][idx]
        claves     = dicc["Claves"][idx]

        nombre_pdf = f"{tt}.pdf"
        existe_pdf = nombre_pdf in pdfs_disponibles

        # Botón “Ver detalles”
        if 'usuario' in session:
            btn_info = (
                f'<button class="btn btn-primary btn-sm" '
                f'data-bs-toggle="modal" data-bs-target="#{modal_id}">'
                'Ver detalles</button>'
            )
        else:
            btn_info = (
                f'<a href="{url_for("login")}" class="btn btn-primary btn-sm">'
                'Ver detalles</a>'
            )

        # Botón “Ver PDF”
        if 'usuario' in session and existe_pdf:
            btn_pdf = (
                f'<a href="/static/pdf/{nombre_pdf}" '
                'target="_blank" class="btn btn-info">Ver PDF</a>'
            )
        elif 'usuario' in session:
            btn_pdf = '<span class="text-muted">PDF no disponible</span>'
        else:
            btn_pdf = (
                f'<a href="{url_for("login")}" '
                'class="btn btn-outline-secondary btn-sm">'
                'Inicia sesión para ver PDF</a>'
            )

        # Fila
        html.append(f'''
        <tr>
          <td>{idx+1}</td>
          <td>{titulo}</td>
          <td>{sim}</td>
          <td>{btn_info}</td>
        </tr>
        ''')

        # Modal detalles (sólo si está logueado)
        if 'usuario' in session:
            info_dir = obtener_info_directores(tt, directores, profesores)
            html.append(f'''
            <div class="modal fade" id="{modal_id}" tabindex="-1" aria-labelledby="label{modal_id}" aria-hidden="true">
              <div class="modal-dialog modal-dialog-scrollable">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title" id="label{modal_id}">{titulo}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                  </div>
                  <div class="modal-body">
                    <p><strong>Resumen:</strong> {resumen}</p>
                    <p><strong>Directores:</strong> {directores}</p>
                    <p><strong>Claves:</strong> {claves}</p>
                    <p><strong>Año:</strong> {tt[:4]}</p>
                  </div>
                  <div class="modal-footer">
                    {btn_pdf}
                    <button class="btn btn-info" data-bs-target="#{modal_id}_2" data-bs-toggle="modal">Directores</button>
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                  </div>
                </div>
              </div>
            </div>
            <div class="modal fade" id="{modal_id}_2" tabindex="-1">
              <div class="modal-dialog modal-dialog-scrollable">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title">Información de los directores</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                  </div>
                  <div class="modal-body">
                    {info_dir}
                  </div>
                  <div class="modal-footer">
                    <button class="btn btn-secondary" data-bs-target="#{modal_id}" data-bs-toggle="modal">Regresar</button>
                  </div>
                </div>
              </div>
            </div>
            ''')

    html.append('''
        </tbody>
      </table>
    </div>
    <p>Esperamos que estos resultados sean de utilidad.</p>
    ''')

    return ''.join(html)


@app.route('/api/buscar', methods=['POST'])
def api_buscar():
    data = request.get_json()

    consulta = data.get('consulta')
    if not consulta:
        return jsonify({"error": "El campo 'consulta' es obligatorio."}), 400

    nombre_profesor = data.get('nombreProfesor') or None
    anio_tt = data.get('anioTT') or None

    resultados = buscarProtocolos(
        consulta=consulta,
        nombreProfesor=nombre_profesor,
        anioTT=anio_tt
    )

    return jsonify(resultados)

#-----------------------------
def obtener_ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def abrir_navegador():
    ip = obtener_ip_local()
    webbrowser.open(f'http://{ip}:5000')

if __name__ == '__main__':
    threading.Timer(1.5, abrir_navegador).start() 
    app.run(host="0.0.0.0", port=5000)


# comando para obtener el ejecutable:
# NOTA: antes debe estar dentro de la carpeta app para que funcione el comando
# python -m PyInstaller --noconfirm --windowed --add-data "templates;templates" --add-data "static;static" --add-data "pkl;pkl" --add-data "protocolos_completo_limpios.csv;." --add-data "profesores_completos_2023.csv;." --add-data "usuarios.db;." pagina.py