# Sistema de Recomendación de Películas

Este proyecto implementa un sistema de recomendación de películas utilizando técnicas de filtrado colaborativo. Proporciona una API basada en FastAPI para agregar usuarios, obtener recomendaciones y recuperar información de usuarios.

# Índice

- [Documentación de RecommendationSystem](#documentación-de-recommendationsystem)
  - [Descripción General](#descripción-general)
  - [Puntos Clave](#puntos-clave)
    - [Carga de Datos Eficiente](#carga-de-datos-eficiente)
    - [Soporte para Usuarios Nuevos](#soporte-para-usuarios-nuevos)
    - [Recomendaciones Personalizadas](#recomendaciones-personalizadas)
      - [Explicación de SVD](#explicación-de-svd)
    - [Manejo del Inicio en Frío](#manejo-del-inicio-en-frío)
    - [Optimización del Rendimiento](#optimización-del-rendimiento)
    - [Flexibilidad](#flexibilidad)
    - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
    - [Manejo de Casos Especiales](#manejo-de-casos-especiales)

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
# Documentación de RecommendationSystem

## Descripción General
`RecommendationSystem` es una clase diseñada para ofrecer un sistema de recomendación de películas eficiente y adaptable. Utiliza técnicas de filtrado colaborativo y maneja casos de inicio en frío para brindar recomendaciones personalizadas.

## Puntos Clave

1. **Carga de Datos Eficiente**:
   - Implementa carga perezosa, cargando datos solo cuando es necesario.
   - Utiliza archivos pickle para deserializar datos rápidamente.

2. **Soporte para Usuarios Nuevos**:
   - Permite agregar usuarios nuevos al sistema de manera dinámica.
   - Realiza validaciones de los datos de entrada para asegurar la integridad.

3. **Recomendaciones Personalizadas**:
   - Usa un algoritmo de filtrado colaborativo para los usuarios ya existentes.
   - Implementa SVD (Descomposición en Valores Singulares) para predicciones más precisas.

### Explicación de SVD:
La Descomposición en Valores Singulares (SVD) es una técnica de factorización matricial utilizada en la recomendación de películas para descomponer la matriz de interacciones usuario-película en tres componentes fundamentales: una matriz de usuarios (`U`), una matriz diagonal de valores singulares (`Σ`), y una matriz de películas (`V^T`). Este enfoque permite reducir la dimensionalidad del problema, extrayendo las características latentes que representan las preferencias tanto de usuarios como de películas. Al multiplicar estas matrices, se pueden reconstruir las interacciones y predecir calificaciones faltantes con gran precisión. Este método es particularmente poderoso cuando se cuenta con datos dispersos, como es el caso típico en sistemas de recomendación.

4. **Manejo del Inicio en Frío**:
   - Ofrece estrategias de recomendación para nuevos usuarios sin historial.
   - Aplica similitud de coseno basada en datos de los usuarios para encontrar usuarios parecidos.

5. **Optimización del Rendimiento**:
   - Utiliza el decorador `@lru_cache` para cachear resultados de recomendaciones frecuentes.
   - Emplea operaciones vectorizadas de NumPy para cálculos más eficientes.

6. **Flexibilidad**:
   - Permite ajustar fácilmente la cantidad de recomendaciones.
   - Proporciona métodos para verificar la existencia de usuarios y obtener las películas que han calificado.

7. **Preprocesamiento de Datos**:
   - Realiza la codificación de variables categóricas como género y ocupación para usarlas en modelos de machine learning.

8. **Manejo de Casos Especiales**:
   - Considera escenarios como usuarios sin películas calificadas o sin usuarios similares.

Esta implementación equilibra la eficiencia computacional con técnicas avanzadas de recomendación, ofreciendo una solución sólida y adaptable para sistemas de recomendación de películas.
