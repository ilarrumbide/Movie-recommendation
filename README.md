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
