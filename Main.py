import numpy as np
from Detector_Objetos import *
import cv2.aruco as aruco


# Cargamos el detector del marcador aruco
parametros = cv2.aruco.DetectorParameters()
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

# Cargamos el detector de objetos
detector = DetectorFondoHomogeneo()

# Realizamos la video captura de nuestra camara
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

# Accedemos al while principal
while True:
    # Realizamos la lectura de la camara
    ret, frame = cap.read()
    if ret == False: break

    # Detectamos el marcador aruco
    esquinas, _, _ = aruco.detectMarkers(frame, diccionario, parameters=parametros)

    # Si se detecta el marcador aruco, continuamos con el proceso
    if esquinas is not None:
        # Solo continuamos si hay al menos un marcador detectado (size > 0)
        if len(esquinas) > 0:
            # Convertimos las esquinas a enteros de 64 bits
            esquinas_ent = np.int64(esquinas[0])  # Access the first detected marker

            # Dibujamos las esquinas del marcador aruco
            cv2.polylines(frame, esquinas_ent, True, (0,0,255), 5)

            # Calculamos el perímetro del marcador aruco
            perimetro_aruco = cv2.arcLength(esquinas_ent, True)

            # Calculamos la proporción en cm
            proporcion_cm = perimetro_aruco / 16

            # Detectamos los objetos en la imagen
            contornos = detector.deteccion_objetos(frame)

            # Dibujamos la detección de los objetos
            for cont in contornos:
                # Dibujamos el contorno del objeto
                # cv2.polylines(frame, [cont], True, (0,255,0), 2)

                # Obtenemos el rectángulo que mejor se ajusta al contorno
                rectangulo = cv2.minAreaRect(cont)
                (x,y), (an,al), angulo = rectangulo

                # Convertimos el ancho y el alto a centímetros
                ancho = an / proporcion_cm
                alto = al / proporcion_cm

                # Dibujamos un círculo en el centro del objeto
                cv2.circle(frame,(int(x), int(y)), 5 , (255,255,0),-1)

                # Obtenemos los puntos del rectángulo
                rect = cv2.boxPoints(rectangulo)
                rect = np.int64(rect)

                # Dibujamos el rectángulo
                cv2.polylines(frame, [rect], True, (0,255,0), 2)

                # Mostramos la información del objeto (ancho y largo en cm)
                cv2.putText(frame, "Ancho: {} cm".format(round(ancho,1)), (int(x), int(y-15)), cv2.LINE_AA, 0.8, (150,0,255),2)
                cv2.putText(frame, "Largo: {} cm".format(round(ancho,1)), (int(x), int(y+15)), cv2.LINE_AA, 0.8, (150,0,255),2)

        else:
            # No se detectó ningún marcador ArUco, puedes mostrar un mensaje informativo
            print("No se detectó ningún marcador ArUco")

    # Mostramos el fotograma
    cv2.imshow('Medición de Objetos', frame)

    # Si se presiona la tecla "q", salimos del programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberamos la captura de la cámara
cap.release()

# Cerramos todas las ventanas
cv2.destroyAllWindows()
