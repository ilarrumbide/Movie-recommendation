# Sistema de Recomendación de Películas

Este proyecto implementa un sistema de recomendación de películas utilizando técnicas de filtrado colaborativo. Proporciona una API basada en FastAPI para agregar usuarios, obtener recomendaciones y recuperar información de usuarios.

## Requisitos previos

Antes de comenzar, asegúrate de cumplir con los siguientes requisitos:

- Tener instalado Python 3.7 o posterior.
- Tener una máquina Windows/Linux/Mac con un compilador C++ instalado (necesario para scikit-surprise).
- Tener pip instalado para gestionar paquetes de Python.

## Configuración del proyecto

Sigue estos pasos para configurar y ejecutar el proyecto:

1. **Clona el repositorio:**

    ```bash
    git clone https://github.com/tuusuario/sistema-recomendacion-peliculas.git
    cd sistema-recomendacion-peliculas
    ```

2. **Crea y activa un entorno virtual (opcional pero recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows, usa `venv\Scripts\activate`
    ```

3. **Instala los paquetes requeridos:**

    ```bash
    pip install -r requirements.txt
    ```

    Nota: Si encuentras problemas al instalar `scikit-surprise`, asegúrate de tener un compilador C++ instalado en tu sistema. Para usuarios de Windows, es posible que necesites instalar Microsoft Visual C++ Build Tools.

4. **Datos y modelo:**

    - Los archivos de datos necesarios ya están incluidos en el directorio `data` del repositorio:
      - `u_user_encoded.pkl`
      - `u_item.pkl`
      - `u_data.pkl`
      - `user_movie_ratings.pkl`

    - El archivo del modelo entrenado `best_svd_model_center.pkl` ya está incluido en el directorio `models` del repositorio.

## Ejecución de la aplicación

Para ejecutar la aplicación, utiliza el siguiente comando:

 ```bash
uvicorn main:app --reload
```
Esto iniciará el servidor FastAPI luego ve a http://127.0.0.1:8000/docs para probrar los Endpoints

## Endpoints de la API

Los siguientes endpoints están disponibles:

### Agregar Usuario

- **URL:** `/add_user`
- **Método:** `POST`
- **Cuerpo:** Objeto JSON con detalles del usuario:
  - `user_id`
  - `age`
  - `gender`
  - `occupation`
  - `zip_code`
- **Respuesta:** Mensaje de confirmación

### Obtener Recomendaciones

- **URL:** `/recommend/{user_id}`
- **Método:** `GET`
- **Respuesta:** Lista de películas recomendadas con calificaciones predichas

### Obtener Información del Usuario

- **URL:** `/information/{user_id}`
- **Método:** `GET`
- **Respuesta:** Lista de películas calificadas por el usuario



Este README proporciona una guía clara para configurar, ejecutar y utilizar el sistema de recomendación.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```bash
sistema-recomendacion-peliculas/
│
├── ml-100k/                     # Directorio con datos del dataset MovieLens 100K
├── models/                      # Directorio para almacenar modelos entrenados
│   └── best_svd_model_center.pkl
│
├── .gitignore                   # Archivo para especificar archivos/directorios ignorados por Git
├── README.md                    # Este archivo
├── analisis_exploratorio.ipynb  # Notebook Jupyter para análisis exploratorio de datos
├── cross_validation_metrics.txt # Métricas de validación cruzada
├── main.py                      # Archivo principal con la implementación de la API FastAPI
├── model.py                     # Implementación del modelo de recomendación
├── no_center_cross_validation_metrics.txt
├── preprocess_data.py           # Script para preprocesamiento de datos
├── requirements.txt             # Lista de dependencias del proyecto
├── test_1m_movies.py            # Script de prueba para 1 millón de películas
├── train.py                     # Script para entrenar el modelo
└── train_center.py              # Script para entrenar el modelo con centralización

```
